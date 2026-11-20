from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
import torch
import jax.numpy as jnp
from flax import nnx
from openpi.vtla.tactile import O10TactileEncoder
from tipdex.forecast.shared_encoder import SharedTactileFeatures, parameter_digest
from tipdex.forecast.model import SharedForecastHead, SharedRuntimeForecaster



def encoder():
    model=SimpleNamespace(tactile_encoder=O10TactileEncoder(256,nnx.Rngs(11),num_channels=364))
    return model,SharedTactileFeatures(model)


def test_shares_actual_object_and_native_preprocessing():
    model,shared=encoder();assert shared.encoder is model.tactile_encoder
    raw=np.arange(16*364,dtype=np.float32).reshape(16,364)
    baseline=np.full((1,1,364),100.,np.float32)
    valid=np.ones((1,16),bool);valid[:,2]=False
    before=parameter_digest(model.tactile_encoder)
    actual=shared.encode(raw,baseline,valid)
    expected=np.asarray(model.tactile_encoder(jnp.asarray(raw[None,None,:,None,:]),
        jnp.asarray(valid),jnp.asarray(baseline[None]),jnp.full((1,1),65535.)))[0]
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
    assert np.all(actual[2]==0) and before==parameter_digest(model.tactile_encoder)
    assert np.max(abs(actual-shared.encode(raw,np.zeros_like(baseline),valid)))>0


def test_shared_encoder_rejects_invalid_registers():
    _,shared=encoder();raw=np.zeros((16,364));raw[0,0]=.5
    with pytest.raises(ValueError,match='raw history'):shared.encode(raw,np.zeros((1,1,364)))


def test_runtime_refuses_encoder_weight_mismatch(tmp_path):
    _,shared=encoder()
    (tmp_path/'contract.json').write_text(json.dumps({'shared_encoder_parameters_sha256':'wrong'}))
    with pytest.raises(ValueError,match='different tactile encoder'):
        SharedRuntimeForecaster(tmp_path,shared,lambda:np.zeros((1,1,364)))


