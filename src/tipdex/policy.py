"""TipDex_VTLA inference with native 9309 and independent VASE/TACE modules."""
import dataclasses
import os
import time

import numpy as np

from openpi.fastslow.policy import FastSlowPolicy
from .sensors import NativeF1Stream
from .schema import ARM_NAMES, HAND_NAMES, IMAGE_KEYS
from .VASE.runtime import VisualReliability


class TipDexPolicy(FastSlowPolicy):
    def __init__(self, source, *, config="paper", revision=None, local_files_only=False):
        from tipdex.bundle import ModelBundle
        from tipdex.config import InferenceConfig
        from tipdex.backbone import load_backbone
        from tipdex.forecast.service import ForecastService
        from tipdex.TACE.runtime import TaceAdapter
        self.config = config if isinstance(config, InferenceConfig) else InferenceConfig.load(config)
        self.bundle = ModelBundle.load(source, revision=revision, local_files_only=local_files_only)
        trained, self._state_normalize = load_backbone(self.bundle, self.config)
        self.metadata = {"prompt": "Move the egg into the box.", "horizon": 30,
                         "state_dim": 19, "physical_action_dim": 13}
        self.vase = VisualReliability(self.bundle.path("VASE"), config=self.config)
        self._vase_features = None
        # Base initialization calls reset(); verifier must already exist.
        self.forecast = ForecastService(self.bundle.path("forecast"), policy_model=trained._model,
                                        baseline_provider=self._forecast_baseline, config=self.config)
        super().__init__(trained, fps=30., max_delay_frames=8, seed=self.config.seed)
        from openpi.shared.nnx_utils import module_jit
        self._slow_context_fn = module_jit(self.model.slow_snapshot_with_context)
        self.slow_fn = self._snapshot_with_context
        self.tace = TaceAdapter(self.bundle.path("TACE"), self.model, self.fast_fn, config=self.config)
        self.fast_fn = self.tace
        self.stream = NativeF1Stream(max_age=.1)
        self.reset()

    def _forecast_baseline(self):
        with self.sensor_lock:
            if self.stream.baseline is None:
                raise ValueError('Shared forecast requires the policy episode calibration')
            return self.stream.baseline.copy()

    def reset(self):
        with self.infer_lock:
            session = super().reset()
            self.vase.reset()
            self.forecast.reset()
            self.forecast.verifier.session = session
            if hasattr(self, 'tace'):
                self.tace.state = self.tace.q = None
            self._vase_features = None
            self._fast_touch_snapshot = None
            return session

    def _snapshot_with_context(self, rng, observation, *, noise=None):
        snapshot, self._vase_features = self._slow_context_fn(rng, observation, noise=noise)
        return snapshot

    def _touch(self, at_time):
        # NativeF1Stream enforces causal timestamps and packet validity.
        # Resample causally; stale native packets are rejected by the stream.
        cached = getattr(self, '_fast_touch_snapshot', None)
        if cached is not None:
            return cached
        with self.sensor_lock:
            fields, stamp = self.stream.snapshot(at_time)
        if not fields['tactile_valid'].any():
            raise ValueError('No reliable current tactile sensors')
        return fields, stamp

    def feed_tactile(self, session_id, raw, valid, timestamp):
        return self.feed_pressure(session_id, raw, valid, timestamp)

    def slow(self, session_id, images, state, timestamp):
        state = np.asarray(state, np.float32)
        if state.shape != (19,) or not np.isfinite(state).all():
            raise ValueError('Expected 19 finite measured state coordinates')
        cameras = {}
        for wire, name in zip(IMAGE_KEYS, ['top', 'left_wrist', 'right_wrist'], strict=True):
            rgb = np.asarray(images[wire])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError('Expected HWC uint8 RGB images')
            cameras[name] = rgb
        with self.infer_lock:
            result = super().slow(session_id, {'images': cameras, 'state': state,
                                             'prompt': self.metadata['prompt']}, timestamp)
            start = time.monotonic()
            visual = self.vase.update(self._vase_features, session_id, result['chunk_id'], timestamp)
            self.forecast.context(session_id, result['chunk_id'], timestamp, self._vase_features)
            vase_ms = (time.monotonic() - start) * 1000
            return {**result, **visual, 'vase_ms': vase_ms, 'slow_ms': result['slow_ms'] + vase_ms}

    def fast(self, session_id, chunk_id, observation_time, executed_steps=0, feedback_state=None, replaceable=None):
        with self.infer_lock:
            self._session(session_id)
            visual = self.vase.response(session_id, chunk_id, observation_time)
            if feedback_state is None:
                raise ValueError('Current state19 is required for TACE')
            if replaceable is not None:
                replaceable = np.asarray(replaceable)
                if replaceable.shape != (30,) or replaceable.dtype != bool:
                    raise ValueError('replaceable must be a boolean mask of length 30')
            self.tace.slot_mask = replaceable
            self.tace.state = None
            self._fast_touch_snapshot = None
            try:
                # One frozen causal snapshot for both error verification and
                # the original policy/refiner, even if the sensor thread advances.
                touch_snapshot = self._touch(observation_time)
                self._fast_touch_snapshot = touch_snapshot
                score = self.forecast.verifier.score(visual.get('q_v'), self.forecast.contract,
                    now=float(observation_time), receipt_time=touch_snapshot[1])
                self.tace.q = score['q_exec']
                if feedback_state is not None:
                    state = np.asarray(feedback_state, np.float32)
                    if state.shape != (19,) or not np.isfinite(state).all():
                        raise ValueError('Expected finite current state19 for TACE')
                    self.tace.state = self._state_normalize({'state':state})['state']
                result = super().fast(session_id, chunk_id, observation_time, executed_steps)
                return {**result, 'action_joint_order': list(ARM_NAMES + HAND_NAMES), **visual,
                        **score,
                        'tace_active':self.tace.params is not None and self.tace.state is not None and self.tace.q is not None}
            finally:
                self._fast_touch_snapshot = None
                self.tace.slot_mask = None

    def feed_frames(self, session_id, frames):
        self._session(session_id)
        for frame in frames:
            self.feed_tactile(session_id, frame['raw364'], frame['valid'], frame['timestamp'])
        self.forecast.verifier.observe(session_id, frames)

    def acknowledge(self, session_id, packets):
        return self.forecast.verifier.acknowledge(session_id, packets)

    def status(self):
        return dict(ready=True, session_id=self.session_id, config=self.config.as_dict(),
                    vase=self.vase.describe(), tace=self.tace.describe(), forecast=self.forecast.describe())
