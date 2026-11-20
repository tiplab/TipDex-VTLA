"""Monotone execution reliability G and explicit no-forecast fallback."""
import math
from tipdex.config import InferenceConfig


def execution_score(q_v, error, *, verified, threshold_low, threshold_full, config=None):
    config = config or InferenceConfig()
    if not all(math.isfinite(float(v)) for v in (q_v, threshold_low, threshold_full)):
        raise ValueError('Nonfinite reliability configuration')
    if not 0 <= q_v <= 1 or not 0 < threshold_low < threshold_full:
        raise ValueError('Invalid score or error thresholds')
    base = config.visual_offset + config.visual_scale * float(q_v)
    if not verified:
        return dict(q_base=base, q_exec=base, tactile_decrement=0., tactile_error=None,
                    tactile_verification='unavailable', fallback='visual_score')
    if error is None or not math.isfinite(float(error)) or error < 0:
        raise ValueError('Verified error must be finite and nonnegative')
    x = min(1., max(0., (error-threshold_low)/(threshold_full-threshold_low)))
    decrement = min(base/2, config.maximum_tactile_decrement*x*x*(3-2*x))
    return dict(q_base=base, q_exec=base-decrement, tactile_decrement=decrement,
                tactile_error=float(error), tactile_verification='verified', fallback=None)
