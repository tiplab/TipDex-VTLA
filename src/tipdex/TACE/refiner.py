"""Differentiable, isolated TACE feature correction for the frozen egg policy.

Public API: ``initialize(...)`` builds a plain dictionary of floating-point JAX
arrays; ``refine_four(...)`` consumes that dictionary and returns normalized
actions [B, 30, 48]. The caller may wrap the latter in ``flax.nnx.jit``. No model
parameters, observations, snapshots, devices, or environment settings are changed.
Only the separate refiner dictionary should be optimized by the caller.

Paper Eq. (4) acts on Action Expert features U, not pooled visual features or
the flow state X. Here U is explicitly the LAST suffix Transformer output,
immediately before action_out_proj. This layer choice and the small GELU MLP
are implementation choices: the paper does not specify that layer or MLP width.
R and S share a trunk taking U, normalized feedback state, and tactile features.
The residual bound is b=0.1 and kappa=1 by default; the execution score is stop-gradient.

The paper's flow time increases from noise at tau=0 to actions at tau=1.
This checkpoint uses reversed time t=1-tau: its six slow steps end near t=0.4,
and the four fast steps integrate by dt=-0.1 to t=0. No teacher-forced bridge
or new noise is substituted for that trajectory. Optional progress has exactly
the stock _denoise(progress) semantics: slots before that cutoff in BOTH
streams are restored to snapshot.x_split after every integration step.

An independent nominal stream uses fresh touch and exactly the stock
cached_velocity. After EVERY step, excluded slots of the corrected stream are
replaced by that step's nominal state. Merely masking the feature residual is
insufficient because subsequent Transformer attention couples action slots.
The corrected stream is never detached: gradients propagate through all four
steps and the frozen suffix. Callers must not wrap that suffix in no_grad.

All feedback must already have the declared normalization and timestamp
alignment. slot_mask is a boolean replaceable/unexecuted mask, not merely a
padding mask. Forecast eligibility and G(q_v,e) belong to the caller. A score
in [0,1] is accepted, with the exact limiting gates at numerical endpoints;
invalid scores or an invalid snapshot phase produce NaNs, never a fabricated
valid action. No implicit visual-score floor or execution-score formula is used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax.nnx.bridge import variables as bridge_variables


ACTION_HORIZON = 30
ACTION_DIM = 48
FEEDBACK_STATE_DIM = 19
FAST_STEPS = 4
FLOW_STEPS = 10
RESIDUAL_BOUND = 0.1
GATE_KAPPA = 1.0


def initialize(key, feature_width: int, state_dim: int, touch_dim: int,
               hidden: int = 64):
    """Create a trainable float32 pytree with explicit input-domain kernels.

    ``u_kernel``, ``state_kernel`` and ``touch_kernel`` have respective shapes
    [feature_width, hidden], [state_dim, hidden], and [touch_dim, hidden]. Their
    projected values are summed before a shared GELU. The residual output
    kernel and bias are exactly zero, so initial corrections are exactly zero.
    Scalar dimensions are encoded by these shapes, not nondifferentiable
    metadata leaves. refine_four currently requires state_dim=19.
    """
    for name, value in (("feature_width", feature_width), ("state_dim", state_dim),
                        ("touch_dim", touch_dim), ("hidden", hidden)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    keys = jax.random.split(key, 3)
    # Equivalent to one Glorot-initialized concatenated input kernel, without
    # materializing a [B,H,state+touch] concatenation at every flow step.
    scale = math.sqrt(2.0 / (feature_width + state_dim + touch_dim + hidden))
    return {
        "u_kernel": jax.random.normal(keys[0], (feature_width, hidden), dtype=jnp.float32) * scale,
        "state_kernel": jax.random.normal(keys[1], (state_dim, hidden), dtype=jnp.float32) * scale,
        "touch_kernel": jax.random.normal(keys[2], (touch_dim, hidden), dtype=jnp.float32) * scale,
        "shared_bias": jnp.zeros((hidden,), dtype=jnp.float32),
        "residual_kernel": jnp.zeros((hidden, feature_width), dtype=jnp.float32),
        "residual_bias": jnp.zeros((feature_width,), dtype=jnp.float32),
        "gate_kernel": jnp.zeros((hidden, 1), dtype=jnp.float32),
        "gate_bias": jnp.zeros((1,), dtype=jnp.float32),
    }


def _parameter_dimensions(params):
    names = {"u_kernel", "state_kernel", "touch_kernel", "shared_bias",
             "residual_kernel", "residual_bias", "gate_kernel", "gate_bias"}
    if not isinstance(params, Mapping) or set(params) != names:
        raise ValueError("params must contain exactly the initialize() parameter keys")
    arrays = {name: jnp.asarray(value) for name, value in params.items()}
    if any(not jnp.issubdtype(value.dtype, jnp.floating) for value in arrays.values()):
        raise ValueError("all refiner parameter leaves must be floating point")
    if any(arrays[name].ndim != 2 for name in ("u_kernel", "state_kernel", "touch_kernel")):
        raise ValueError("input-domain kernels must be matrices")
    width, hidden = arrays["u_kernel"].shape
    state_dim = arrays["state_kernel"].shape[0]
    touch_dim = arrays["touch_kernel"].shape[0]
    if min(width, hidden, state_dim, touch_dim) < 1:
        raise ValueError("refiner dimensions must be positive")
    shapes = {"state_kernel": (state_dim, hidden), "touch_kernel": (touch_dim, hidden),
              "shared_bias": (hidden,), "residual_kernel": (hidden, width),
              "residual_bias": (width,), "gate_kernel": (hidden, 1), "gate_bias": (1,)}
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"invalid parameter shape for {name}: expected {shape}")
    return width, state_dim, touch_dim


def _make_attn_mask(input_mask, mask_ar):
    """The same prefix/suffix attention rule as the owning Pi0 implementation."""
    cumulative = jnp.cumsum(jnp.broadcast_to(mask_ar, input_mask.shape), axis=1)
    order = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(order, valid)


def _suffix_features(model, snapshot, observation, x_t, time, condition):
    """Match Pi05FastSlow.cached_velocity up to, but excluding, action projection."""
    batch = x_t.shape[0]
    tokens, suffix_mask, ar_mask, adarms = model.embed_suffix(
        observation, x_t, jnp.broadcast_to(time, (batch,)), condition)
    prefix_attn = jnp.broadcast_to(
        snapshot.prefix_mask[:, None, :],
        (batch, tokens.shape[1], snapshot.prefix_mask.shape[1]))
    attn = jnp.concatenate([prefix_attn, _make_attn_mask(suffix_mask, ar_mask)], axis=-1)
    positions = (jnp.sum(snapshot.prefix_mask, axis=-1)[:, None]
                 + jnp.cumsum(suffix_mask, axis=-1) - 1)
    # Evaluate the two correction streams in FP32 without changing base weights,
    # the native forward path or the cached prefix values. This reduces BF16
    # threshold crossings caused by a very small learned feature residual.
    llm = model.PaliGemma.llm
    variables = bridge_variables.nnx_attrs_to_linen_vars(
        {name: getattr(llm, name) for name in llm.linen_attributes})
    (_, output), _ = llm.module.clone(embed_dtype='float32').apply(
        variables, [None, tokens.astype(jnp.float32)], mask=attn, positions=positions,
        kv_cache=jax.tree.map(lambda x: x.astype(jnp.float32), snapshot.kv_cache),
        adarms_cond=[None, adarms])
    return output[:, -model.action_horizon:]


def _correct_features(features, feedback_state, feedback_touch, q, slot_mask, params, *, bound=RESIDUAL_BOUND, kappa=GATE_KAPPA):
    inputs = (features.astype(jnp.float32) @ params["u_kernel"]
              + (feedback_state @ params["state_kernel"])[:, None, :]
              + (feedback_touch @ params["touch_kernel"])[:, None, :]
              + params["shared_bias"])
    shared = jax.nn.gelu(inputs, approximate=False)
    residual = shared @ params["residual_kernel"] + params["residual_bias"]
    learned_logit = shared @ params["gate_kernel"] + params["gate_bias"]
    score = jax.lax.stop_gradient(q)
    # No clipping/remapping: endpoints give their limiting gates, and invalid
    # scores propagate NaN. stop_gradient prevents L_R from updating VASE/G.
    score_logit = jnp.log(score) - jnp.log1p(-score)
    gate = jax.nn.sigmoid(learned_logit - kappa * score_logit[:, None, None])
    score_valid = jnp.isfinite(score) & (score >= 0) & (score <= 1)
    gate = jnp.where(score_valid[:, None, None], gate, jnp.nan)
    delta = bound * gate * jnp.tanh(residual)
    # Return the float32 perturbation; the caller projects delta through the
    # frozen linear kernel and anchors the trajectory delta to native stock.
    return jnp.where(slot_mask[..., None], delta.astype(jnp.float32), 0.)


def _integrate_streams(nominal_x, corrected_x, nominal_velocity,
                       corrected_velocity, slot_mask, dt, *, x_split=None,
                       progress=None):
    """Restore excluded slots to nominal, with stock progress clamping if set."""
    next_nominal = nominal_x + dt * nominal_velocity
    next_corrected = corrected_x + dt * corrected_velocity
    if progress is not None:
        future = jnp.arange(slot_mask.shape[1])[None, :] >= progress[:, None]
        next_nominal = jnp.where(future[..., None], next_nominal, x_split)
        slot_mask = slot_mask & future
    return next_nominal, jnp.where(slot_mask[..., None], next_corrected, next_nominal)


def refine_four(model, snapshot, observation, feedback_state, feedback_touch,
                q, slot_mask, params, *, progress=None, return_delta=False, bound=RESIDUAL_BOUND, kappa=GATE_KAPPA):
    """Return normalized [B,30,48] actions after four differentiable fast steps.

    With return_delta=True, return corrected-minus-nominal from the same graph;
    the caller adds this to its separately compiled native stock trajectory.

    ``snapshot`` is the frozen perturbed-branch nominal slow snapshot. The
    observation supplies fresh tactile conditioning to BOTH streams, matching
    stock ``fast_refine(snapshot, observation, progress=progress)``. Explicit
    feedback_state [B,19] and feedback_touch [B,D] condition only the refiner.
    q [B] is the already-computed execution score; slot_mask [B,30] must be bool.
    Optional progress is integer [B] in [0,30]. Earlier slots of BOTH streams
    are clamped to snapshot.x_split at every step, even if slot_mask is True.
    With progress=None, the original unclamped nominal stream is preserved.
    The caller owns normalization, execution eligibility, and frozen-base
    parameter handling. This function does not generate or validate a forecast.
    """
    width, state_dim, touch_dim = _parameter_dimensions(params)
    if (model.action_horizon != ACTION_HORIZON or model.action_dim != ACTION_DIM
            or model.flow_steps != FLOW_STEPS or model.slow_steps != FLOW_STEPS - FAST_STEPS):
        raise ValueError("refine_four requires horizon=30, action_dim=48 and 6+4/10 flow steps")
    if state_dim != FEEDBACK_STATE_DIM:
        raise ValueError("the egg refiner requires feedback_state dimension 19")
    x = jnp.asarray(snapshot.x_split)
    if x.ndim != 3 or x.shape[1:] != (ACTION_HORIZON, ACTION_DIM) or x.shape[0] < 1:
        raise ValueError("snapshot.x_split must have shape [B,30,48] with B>0")
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise ValueError("snapshot.x_split must be floating point")
    batch = x.shape[0]
    feedback_state = jnp.asarray(feedback_state, dtype=jnp.float32)
    feedback_touch = jnp.asarray(feedback_touch, dtype=jnp.float32)
    q = jnp.asarray(q, dtype=jnp.float32)
    slot_mask = jnp.asarray(slot_mask)
    if (feedback_state.shape != (batch, state_dim)
            or feedback_touch.shape != (batch, touch_dim) or q.shape != (batch,)):
        raise ValueError("feedback_state, feedback_touch or q has an incompatible shape")
    if slot_mask.shape != (batch, ACTION_HORIZON) or slot_mask.dtype != jnp.bool_:
        raise ValueError("slot_mask must be boolean [B,30]")
    if progress is not None:
        progress = jnp.asarray(progress)
        if progress.shape != (batch,) or not jnp.issubdtype(progress.dtype, jnp.integer):
            raise ValueError("progress must be integer [B]")
        future = jnp.arange(ACTION_HORIZON)[None, :] >= progress[:, None]
        slot_mask = slot_mask & future
    if jnp.shape(snapshot.time) != ():
        raise ValueError("snapshot.time must be a scalar in reversed flow time")
    if snapshot.prefix_mask.ndim != 2 or snapshot.prefix_mask.shape[0] != batch:
        raise ValueError("snapshot.prefix_mask must have shape [B,prefix_length]")
    params = {name: jnp.asarray(value, dtype=jnp.float32) for name, value in params.items()}
    condition = model.encode_tactile_condition(observation)
    dt = -1.0 / model.flow_steps

    def step(_, carry):
        nominal_x, corrected_x, time = carry
        # A single batched graph handles both streams: zero residual must not
        # compare two differently fused/rounded compilations of the suffix.
        both_features = jax.vmap(lambda stream_x: _suffix_features(
            model, snapshot, observation, stream_x, time, condition))(
                jnp.stack((nominal_x, corrected_x)))
        velocities = model.action_out_proj(both_features).astype(jnp.float32)
        nominal_velocity = velocities[0]
        features = both_features[1]
        if features.shape != (batch, ACTION_HORIZON, width):
            raise ValueError("feature_width differs from the final Action Expert suffix output")
        feature_delta = _correct_features(
            features, feedback_state, feedback_touch, q, slot_mask, params, bound=bound, kappa=kappa)
        base_velocity = velocities[1]
        # Linear(U + delta) - Linear(U) = delta @ W. Evaluate the
        # small correction in float32 so BF16 rounding cannot erase it.
        correction_velocity = feature_delta @ model.action_out_proj.kernel.value.astype(jnp.float32)
        corrected_velocity = base_velocity + correction_velocity
        nominal_next, corrected_next = _integrate_streams(
            nominal_x, corrected_x, nominal_velocity, corrected_velocity, slot_mask, dt,
            x_split=x, progress=progress)
        return nominal_next, corrected_next, time + dt

    nominal_actions, actions, _ = jax.lax.fori_loop(0, FAST_STEPS, step, (x, x, snapshot.time))
    if return_delta:
        # Anchor this mathematically equivalent delta to the separately compiled
        # native stock result in the caller. This excludes cross-graph BF16
        # compilation drift from both the correction and its ablation.
        actions = actions - nominal_actions
    phase_valid = (jnp.isfinite(snapshot.time)
                   & jnp.isclose(snapshot.time, 0.4, rtol=0, atol=2e-6))
    if progress is not None:
        valid_progress = (progress >= 0) & (progress <= ACTION_HORIZON)
        actions = jnp.where(valid_progress[:, None, None], actions, jnp.full_like(actions, jnp.nan))
    return jnp.where(phase_valid, actions, jnp.full_like(actions, jnp.nan))
