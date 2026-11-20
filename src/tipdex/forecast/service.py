"""CPU forecasting side channel, never a robot command endpoint."""
from collections import deque
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np


class ForecastService:
    def __init__(self, directory, *, policy_model, baseline_provider, config, clock=time.monotonic):
        self.clock = clock
        from .verifier import ForecastVerifier
        from .shared_encoder import SharedTactileFeatures
        from .model import SharedRuntimeForecaster
        import torch
        torch.set_num_threads(1)
        self.directory = Path(directory)
        self.contexts = deque(maxlen=8)
        self.context_lock = threading.RLock()
        self.lock = threading.Lock()
        self.config = config
        self.verifier = ForecastVerifier(config=config)
        self.contract = json.loads((self.directory/'contract.json').read_text())
        if self.contract.get('architecture') != 'shared_9309_recurrent_forecaster_v1':
            raise ValueError('Unknown forecast decoder architecture')
        shared = SharedTactileFeatures(policy_model)
        self.model = SharedRuntimeForecaster(self.directory, shared, baseline_provider,
                                            horizon=config.forecast_horizon)
        self.reason = 'ready'

    def describe(self):
        return dict(enabled=True, status='ready', horizon=self.config.forecast_horizon,
                    shape=[self.config.forecast_horizon,14,26],
                    layout='flat_register_order_14x26', anatomical_mapping=False,
                    units='raw_uint16_divided_by_65535', shared_tactile_encoder=True,
                    verification='latest_eligible_prediction_for_current_frame',
                    trigger='accepted_action_plan_update', error_aggregation='current_frame_mae')

    def reset(self):
        with self.context_lock:
            self.contexts.clear()
        self.verifier.reset()

    def context(self, session, chunk, stamp, features):
        if self.model is not None:
            value = np.array(features, np.float32).reshape(2048)
            value.setflags(write=False)
            with self.context_lock:
                self.contexts.append((session, chunk, float(stamp), value))

    def predict(self, request):
        if self.model is None:
            raise ValueError('Forecaster unavailable: '+self.reason)
        import copy
        request = copy.deepcopy(request)
        # One atomic deque snapshot; do not contend with JAX's inference lock.
        with self.context_lock:
            contexts = list(self.contexts)
        found = next((row for row in reversed(contexts)
                      if row[:2] == (request['session_id'], request['chunk_id'])), None)
        if found is None:
            raise ValueError('Forecast context expired or session mismatch')
        anchor = float(request['anchor_time'])
        history_times = np.asarray(request['history_times'], np.float64)
        target_times = np.asarray(request['target_times'], np.float64)
        start = self.clock()
        if not np.isfinite(anchor) or not 0 <= anchor-found[2] <= .3:
            raise ValueError('Forecast context is not causal/recent')
        if not 0 <= start-anchor <= .1:
            raise ValueError('Forecast anchor is not fresh on the local monotonic clock')
        with self.lock:
            prediction = self.model.predict(found[3], request['touch_history'],
                request['state_history'], request['command_history'], history_times-anchor,
                request['future_commands'], target_times-anchor)
        issued = self.clock()
        if request['session_id'] != self.verifier.session:
            raise ValueError('Session changed during prediction')
        if np.any(target_times <= issued):
            raise ValueError('Forecast missed a target time; not issuing a retrospective prediction')
        result = dict(forecast_id=uuid.uuid4().hex, session_id=request['session_id'],
                    chunk_id=request['chunk_id'], plan_id=request['plan_id'],
                    anchor_time=anchor, issued_at=issued, target_times=target_times.tolist(),
                    predicted_normalized=prediction.tolist(),
                    future_at_issuance=(target_times>issued).tolist(),
                    prediction_ms=(issued-start)*1000,
                    layout='flat_register_order_14x26', anatomical_mapping=False,
                    normalization_id='raw_uint16_divided_by_65535',
                    shared_tactile_encoder=True,
                    verification='pending_actual_command_and_measurement_alignment')

        self.verifier.register(request, result)
        return result
