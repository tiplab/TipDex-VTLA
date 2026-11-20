"""Learned feature residuals on the four remaining flow steps."""
import json
from pathlib import Path
import numpy as np
from tipdex.bundle import sha256


class TaceAdapter:
    def __init__(self, directory, model, native_fast, *, config):
        import jax
        import jax.numpy as jnp
        from flax import nnx, serialization
        from .refiner import initialize, refine_four
        root = Path(directory)
        meta = json.loads((root/'config.json').read_text())
        if meta['architecture'] != 'shared_gelu_feature_residual_v1':
            raise ValueError('Unknown TACE architecture')
        if sha256(root/'tace.msgpack') != meta['weights_sha256']:
            raise ValueError('TACE weight hash mismatch')
        self.model, self.native_fast, self.config = model, native_fast, config
        self.state = self.q = self.slot_mask = None
        self.params = serialization.from_bytes(initialize(jax.random.key(0), *meta['template_dims']),
                                               (root/'tace.msgpack').read_bytes())
        if not all(np.isfinite(np.asarray(v)).all() for v in self.params.values()):
            raise ValueError('Nonfinite TACE weights')
        bound, kappa = config.residual_bound, config.gate_kappa

        @nnx.jit
        def delta(m, params, snapshot, observation, state, q, progress, mask):
            ft, valid, _, _ = m.encode_tactile_condition(observation)
            ft = jnp.where(valid[...,None],ft,0).astype(jnp.float32).sum(1)/jnp.maximum(valid.sum(1,keepdims=True),1)
            mask = mask & (jnp.arange(m.action_horizon)[None,:] >= progress[:,None])
            return refine_four(m,snapshot,observation,state,ft,q,mask,params,
                               progress=progress,return_delta=True,bound=bound,kappa=kappa)
        self.delta = delta

    def describe(self):
        return dict(enabled=True, status='ready', feature_bound=self.config.residual_bound,
                    gate_kappa=self.config.gate_kappa, protected_slots='never modified',
                    score='G(q_v,e)', correction='action expert features at each fast flow step')

    def __call__(self, snapshot, observation, progress):
        import jax.numpy as jnp
        actions, gate = self.native_fast(snapshot, observation, progress)
        if self.state is None or self.q is None:
            raise ValueError('TACE requires current normalized state and reliability')
        if not np.isfinite(self.q) or not 0 <= self.q <= 1:
            raise ValueError('Invalid TACE reliability')
        mask = jnp.ones(actions.shape[:2],bool) if self.slot_mask is None else jnp.asarray(self.slot_mask)[None]
        delta = self.delta(self.model,self.params,snapshot,observation,
            jnp.asarray(self.state,jnp.float32)[None],jnp.asarray([self.q],jnp.float32),progress,mask)
        mask = mask & (jnp.arange(actions.shape[1])[None,:] >= progress[:,None])
        return jnp.where(mask[...,None],actions+delta,actions), gate
