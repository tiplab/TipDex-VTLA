"""Read-only access to the policy's actual tactile encoder, not a copied model."""
import hashlib
import json
import numpy as np


def parameter_digest(encoder):
    from flax import nnx
    import jax
    state = nnx.state(encoder, nnx.Param).to_pure_dict()
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        array = np.asarray(value)
        digest.update(json.dumps([str(path), str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class SharedTactileFeatures:
    def __init__(self, model):
        from flax import nnx
        self.encoder = model.tactile_encoder
        self.parameters_sha256 = parameter_digest(self.encoder)

        @nnx.jit
        def encode(encoder, history, valid, baseline, scale):
            return encoder(history, valid, baseline, scale)

        self._encode = encode
        # Compile the singleton runtime shape before accepting timed forecasts.
        self.encode(np.zeros((16, 364), np.float32), np.zeros((1, 1, 364), np.float32))

    def encode(self, history, baseline, valid=None):
        import jax.numpy as jnp
        history = np.asarray(history, np.float32)
        baseline = np.asarray(baseline, np.float32)
        singleton = history.ndim == 2
        if singleton:
            history = history[None]
            baseline = baseline.reshape(1, 1, 1, 364)
        n = len(history)
        valid = np.ones((n,16), bool) if valid is None else np.asarray(valid, bool).reshape(n,16)
        if (history.shape != (n,16,364) or baseline.shape != (n,1,1,364)
            or not np.isfinite(history).all() or not np.isfinite(baseline).all()
            or np.any((history < 0) | (history > 65535))
            or np.any(history != np.floor(history))):
            raise ValueError('Invalid shared-encoder raw history or episode baseline')
        result = np.asarray(self._encode(self.encoder, jnp.asarray(history[:,None,:,None,:]),
            jnp.asarray(valid), jnp.asarray(baseline), jnp.full((n,1),65535.,jnp.float32)), np.float32)
        if result.shape != (n,16,256) or not np.isfinite(result).all():
            raise ValueError('Invalid shared tactile features')
        return result[0] if singleton else result
