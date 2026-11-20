"""Paper Sec. 3.3: latest eligible prediction for the current tactile frame.

A forecast is an immutable, pre-measurement statement about a timestamped
command schedule. Only its prefix through the measured frame is verified.
Unobserved later frames never delay or invalidate an already eligible frame.
SDK ACKs certify accepted commands, not physical pose or task success.
"""
from collections import deque
import copy
import threading
import time
import numpy as np
from .execution import ExecutionEvidence
from tipdex.TACE.score import execution_score

NORMALIZATION = 'raw_uint16_divided_by_65535'


class ForecastVerifier:
    def __init__(self, host_boot_id=None, *, config=None):
        from tipdex.config import InferenceConfig
        self.config = config or InferenceConfig()
        self.lock = threading.RLock()
        self.execution = ExecutionEvidence(host_boot_id)
        self.reset()

    def reset(self):
        with self.lock:
            self.session = None
            self.pending = deque(maxlen=128)
            self.measurements = deque(maxlen=512)
            self.execution.reset()
            self.last_reason = 'no_eligible_forecast_for_current_frame'

    def acknowledge(self, session, packets):
        with self.lock:
            if session != self.session or len(packets) > 1024:
                raise ValueError('Invalid execution ACK session or batch')
            for packet in packets:
                self.execution.ingest(packet)
            return self.execution.describe()

    def observe(self, session, frames, now=None):
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            if session != self.session:
                return
            for frame in frames:
                stamp = frame.get('acquisition_time')
                raw = np.asarray(frame.get('raw364', frame.get('raw')), np.float64)
                valid = np.asarray(frame.get('valid'), bool)
                receipt = frame.get('timestamp', stamp)
                if (stamp is not None and receipt is not None and np.isfinite([stamp,receipt]).all() and 0 <= stamp <= receipt <= now
                    and raw.shape == (364,) and np.isfinite(raw).all()
                    and np.all((raw >= 0) & (raw <= 65535))
                    and np.all(raw == np.floor(raw)) and valid.size and valid.all()):
                    if not self.measurements or stamp > self.measurements[-1][0]:
                        value = np.array(raw / 65535., copy=True)
                        value.setflags(write=False)
                        self.measurements.append((float(stamp), value, float(receipt)))

    def register(self, request, result):
        with self.lock:
            if request['session_id'] != self.session or result['session_id'] != self.session:
                raise ValueError('Forecast verifier session mismatch')
            times = np.asarray(result['target_times'], np.float64)
            prediction = np.asarray(result['predicted_normalized'], np.float64)
            issued = float(result['issued_at'])
            anchor = float(request['anchor_time'])
            history = np.asarray(request['history_times'], np.float64)
            commands = np.asarray(request['command_history'], np.float64)
            future = np.asarray(request['future_commands'], np.float64)
            if (not np.isfinite(issued) or not np.isfinite(anchor) or anchor > issued
                or times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or np.any(times <= issued)
                or prediction.shape != (len(times), 364) or not np.isfinite(prediction).all()
                or np.any((prediction < 0) | (prediction > 1))
                or history.ndim != 1 or not len(history) or not np.isfinite(history).all()
                or np.any(np.diff(history) <= 0) or history[-1] != anchor
                or commands.shape != (len(history), 13) or not np.isfinite(commands).all()
                or future.shape != (len(times), 13) or not np.isfinite(future).all()
                or not np.array_equal(times, request['target_times'])
                or result.get('normalization_id') != NORMALIZATION
                or result['plan_id'] != request['plan_id']):
                raise ValueError('Invalid causal forecast contract')
            schedule_times = np.asarray(request.get('command_times', [anchor]+times.tolist()), np.float64)
            schedule_values = np.asarray(request.get('scheduled_commands', [commands[-1].tolist()]+future.tolist()), np.float64)
            if (schedule_times.ndim != 1 or not len(schedule_times)
                or schedule_values.shape != (len(schedule_times),13)
                or not np.isfinite(schedule_times).all() or not np.isfinite(schedule_values).all()
                or schedule_times[0] != anchor or np.any(np.diff(schedule_times)<=0)):
                raise ValueError('Invalid exact conditioned command schedule')
            sampled = schedule_values[np.searchsorted(schedule_times, times, side='right')-1]
            if not np.allclose(sampled, future, atol=1.e-6, rtol=0):
                raise ValueError('Forecast samples differ from conditioned command schedule')
            # Concurrent callers cannot modify a prediction after issuance.
            self.pending.append((copy.deepcopy(request), copy.deepcopy(result)))

    def score(self, q_v, contract, now=None, measurement_time=None, receipt_time=None):
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError('Invalid verification clock')
        if q_v is None:
            return dict(q_base=None, q_exec=None, tactile_error=None, tactile_decrement=0.,
                        tactile_verification='unavailable', tactile_reason='q_v_unavailable',
                        tactile_error_for_gate=0.)
        low = contract.get('error_threshold_low', 1.)
        full = contract.get('error_threshold_full', 2.)
        tolerance = float(contract.get('alignment_tolerance_seconds', 1/60))
        max_age = float(contract.get('measurement_max_age_seconds', .05))
        if not np.isfinite([tolerance, max_age]).all() or not 0 <= tolerance <= max_age:
            raise ValueError('Invalid measurement alignment configuration')
        unavailable = execution_score(q_v, None, verified=False,
            threshold_low=low, threshold_full=full, config=self.config)
        with self.lock:
            # Never select an old convenient measurement or a future sample.
            rows = [r for r in self.measurements if r[0] <= now and r[2] <= now]
            row = rows[-1] if rows else None
            reason = 'no_current_tactile_measurement'
            if measurement_time is not None:
                row = next((r for r in reversed(rows) if abs(r[0]-measurement_time) < 1e-8), None)
            if receipt_time is not None:
                row = next((r for r in reversed(rows) if abs(r[2]-receipt_time) < 1e-8), None)
            if row is not None and now-row[0] > max_age:
                row, reason = None, 'current_tactile_measurement_stale'
            if row is not None:
                stamp, actual, receipt = row
                reason = 'no_eligible_forecast_for_current_frame'
                # Issuance order, not HTTP completion or registration order.
                for request, result in sorted(self.pending, key=lambda r:r[1]['issued_at'], reverse=True):
                    times = np.asarray(result['target_times'])
                    index = int(np.argmin(np.abs(times-stamp)))
                    if abs(times[index]-stamp) > tolerance or result['issued_at'] >= stamp:
                        continue
                    if not self.execution.history_matches(request['history_times'], request['command_history']):
                        reason = 'command_history_not_verified'
                        continue
                    matched, reason = self.execution.schedule_matches(request['anchor_time'], stamp,
                        request.get('command_times', [request['anchor_time']] + request['target_times']),
                        request.get('scheduled_commands', [request['command_history'][-1]] + request['future_commands']))
                    if not matched:
                        continue
                    error = float(np.abs(np.asarray(result['predicted_normalized'])[index]-actual).mean())
                    score = execution_score(q_v, error, verified=True,
                        threshold_low=low, threshold_full=full, config=self.config)
                    self.last_reason = reason
                    return dict(**score, forecast_id=result['forecast_id'], plan_id=result['plan_id'],
                        forecast_issued_at=result['issued_at'], forecast_target_time=float(times[index]),
                        forecast_frame_index=index, tactile_measurement_time=stamp, tactile_receipt_time=receipt,
                        tactile_error_for_gate=error, tactile_reason=reason,
                        tactile_error_age_s=now-stamp,
                        tactile_error_aggregation='current_frame_mae_over_364_values')
            self.last_reason = self.execution.failure or reason
            return dict(**unavailable, tactile_reason=self.last_reason, tactile_error_for_gate=0.)
