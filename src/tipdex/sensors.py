"""Causal raw F1 stream for demo01 EGG; no model or hardware imports."""
import math
from collections import deque
import numpy as np


class NativeF1Stream:
    """Training-rate causal uint16 registers, first-16 median, scale 65535."""
    def __init__(self, fps=30., history=16, max_age=None):
        self.fps, self.history, self.max_age = float(fps), history, max_age
        self.reset()

    def reset(self):
        self.frames, self.valid, self.times = deque(), deque(), deque()
        self.initial, self.initial_valid, self.initial_times = [], [], []
        self.baseline = None
        self.calibration_time = math.inf
        self.calibration_targets = None

    def set_calibration_targets(self, timestamps):
        targets = np.asarray(timestamps, np.float64)
        if (self.times or targets.shape != (16,) or not np.isfinite(targets).all()
                or np.any(np.diff(targets) <= 0) or np.any(np.diff(targets) > .1)):
            raise ValueError('Set 16 increasing top-camera calibration anchors before tactile feed')
        self.calibration_targets = targets.copy()

    def _sample(self, frames, valid, times, targets):
        times = np.asarray(times, np.float64)
        indices = np.searchsorted(times, targets + 1e-9, side='right') - 1
        chosen = indices.clip(min=0)
        age = targets - times[chosen]
        fresh = (indices >= 0) & (age >= -1e-9)
        if self.max_age is not None:
            fresh &= age <= self.max_age + 1e-9
        validity = np.asarray(valid, bool)[chosen] & fresh
        values = np.asarray(frames, np.float32)[chosen]
        return np.where(validity[:, None], values, 0), validity

    def update(self, raw, valid, timestamp):
        raw, timestamp = np.asarray(raw), float(timestamp)
        if raw.shape != (364,) or not np.isfinite(raw).all():
            raise ValueError('Expected 364 finite F1 register values')
        if np.any(raw < 0) or np.any(raw > 65535) or np.any(raw != np.floor(raw)):
            raise ValueError('F1 registers must be lossless uint16 values')
        if not np.isfinite(timestamp) or (self.times and timestamp <= self.times[-1]):
            raise ValueError('Tactile timestamps must strictly increase')
        if np.asarray(valid).size != 1:
            raise ValueError('Expected one packet-valid flag')
        valid = bool(np.asarray(valid).item())
        raw = raw.astype(np.float32, copy=True)
        self.frames.append(raw); self.valid.append(valid); self.times.append(timestamp)
        while len(self.times) > 2 and self.times[1] < timestamp - 5.:
            self.frames.popleft(); self.valid.popleft(); self.times.popleft()
        if self.baseline is None:
            self.initial.append(raw); self.initial_valid.append(valid); self.initial_times.append(timestamp)
            targets = (self.initial_times[0] + np.arange(16) / self.fps
                       if self.calibration_targets is None else self.calibration_targets)
            if timestamp >= targets[-1] - 1e-9:
                values, validity = self._sample(self.initial, self.initial_valid, self.initial_times, targets)
                self.baseline = (np.median(values[validity], axis=0) if validity.any()
                                 else np.zeros(364, np.float32))[None, None, :]
                self.calibration_time = float(targets[-1])
                self.initial, self.initial_valid, self.initial_times = [], [], []

    def snapshot(self, at_time):
        at_time = float(at_time)
        if not np.isfinite(at_time):
            raise ValueError('Observation timestamp must be finite')
        if self.baseline is None or at_time + 1e-9 < self.calibration_time:
            raise ValueError('Feed the first 16 training-rate tactile anchors before inference')
        times = np.asarray(self.times, np.float64)
        end = int(np.searchsorted(times, at_time + 1e-9, side='right'))
        if end == 0:
            raise ValueError('No causal tactile packet')
        targets = at_time + np.arange(1 - self.history, 1) / self.fps
        raw, valid = self._sample(list(self.frames)[:end], list(self.valid)[:end], times[:end], targets)
        return {
            'tactile': raw[None, :, None, :],
            'tactile_valid': valid[-1:].copy(),
            'tactile_frame_valid': valid,
            'tactile_sample_valid': valid[None, :],
            'tactile_baseline': self.baseline.copy(),
            'tactile_scale': np.full((1,), 65535., np.float32),
            'tactile_contact': np.asarray(0., np.float32),
            'tactile_contact_valid': np.asarray(False, bool),
        }, float(times[end - 1])

