from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import torch
from tipdex.config import InferenceConfig
from tipdex.TACE.score import execution_score
from tipdex.TACE.refiner import initialize, _correct_features, _integrate_streams
from tipdex.forecast.model import SharedForecastHead


def score(q,e,profile='paper',verified=True):
    return execution_score(q,e,verified=verified,threshold_low=.005,threshold_full=.01,
                           config=InferenceConfig.preset(profile))['q_exec']


def test_paper_score_identity_monotonicity_and_large_error_only():
    for q in np.linspace(0,1,31):
        assert score(q,0)==q
        assert score(q,None,verified=False)==q
        assert score(q,.004)==q
        values=[score(q,e) for e in np.linspace(0,.1,50)]
        assert np.all(np.diff(values)<=1e-14)
        assert 0 <= q-values[-1] <= .10000001
    assert np.all(np.diff([score(q,1) for q in np.linspace(0,1,100)])>=0)
    assert score(.8,0,'tipdex')==pytest.approx(.92)
    assert score(.8,1,'tipdex')==pytest.approx(.82)


def test_paper_profile_rejects_silent_overrides():
    with pytest.raises(ValueError):InferenceConfig(visual_offset=.6,visual_scale=.4)
    with pytest.raises(ValueError):InferenceConfig(forecast_horizon=5)
    with pytest.raises(ValueError):InferenceConfig.from_dict({'unknown':1})
    with pytest.raises(ValueError):InferenceConfig(profile='custom',residual_bound=float('nan'))


def test_feature_gate_bound_mask_and_stop_gradient():
    params=initialize(jax.random.key(3),8,19,256)
    params['residual_bias']=jnp.ones(8)*2
    features=jnp.ones((1,30,8));state=jnp.ones((1,19));touch=jnp.ones((1,256))
    mask=jnp.ones((1,30),bool).at[:,:3].set(False).at[:,8].set(False)
    def run(q,b=.1):return _correct_features(features,state,touch,q,mask,params,bound=b)
    low=np.asarray(run(jnp.array([.2])));high=np.asarray(run(jnp.array([.9])))
    assert np.max(abs(low)) <= .1 and np.max(abs(low))>np.max(abs(high))
    assert np.all(low[:,[0,1,2,8]]==0)
    np.testing.assert_array_equal(jax.grad(lambda q:jnp.sum(run(q)))(jnp.array([.5])),[0])
    np.testing.assert_allclose(run(jnp.array([.2]),.01),low*.1,atol=1e-8)


def test_protected_slots_restored_at_each_integration_step():
    mask=jnp.array([[False,True,False]])
    nominal,corrected=_integrate_streams(jnp.zeros((1,3,2)),jnp.ones((1,3,2)),
        jnp.ones((1,3,2)),jnp.ones((1,3,2))*7,mask,-.1)
    np.testing.assert_array_equal(np.asarray(corrected)[:,[0,2]],np.asarray(nominal)[:,[0,2]])


def test_recurrent_forecast_supports_H_and_keeps_five_frame_prefix():
    torch.manual_seed(0);model=SharedForecastHead().eval()
    torch.nn.init.normal_(model.output.weight,std=.001)
    common=[torch.randn(1,2048),torch.randn(1,16,256),torch.randn(1,16,19),
            torch.randn(1,16,13),torch.arange(-15,1)[None]/30.]
    commands=torch.randn(1,30,13);times=torch.arange(1,31)[None]/30.
    with torch.inference_mode():
        full=model(*common,commands,times,torch.full((1,364),.5),torch.ones(1,364))
        short=model(*common,commands[:,:5],times[:,:5],torch.full((1,364),.5),torch.ones(1,364))
    assert full.shape==(1,30,364) and torch.isfinite(full).all()
    torch.testing.assert_close(full[:,:5],short)
    assert not any('encoder' in name for name,_ in model.named_parameters())


def test_context_concatenation_is_separate_from_action_tokens():
    from openpi.models.pi0 import Pi0,make_attn_mask
    class Projection:
        out_features=8
        def __call__(self,x):return jnp.broadcast_to(x.mean(-1,keepdims=True),(*x.shape[:-1],8))
    fake=SimpleNamespace(continuous_state_input=False,pi05=True,action_horizon=30,
        action_in_proj=Projection(),time_mlp_in=lambda x:x,time_mlp_out=lambda x:x,
        tactile_fusion='context_concat',tactile_adapter=SimpleNamespace(value=lambda x:x,output=Projection()))
    from openpi.models.model import Observation
    obs=Observation(images={},image_masks={},state=jnp.zeros((1,48)))
    condition=(jnp.ones((1,16,256)),jnp.ones((1,16),bool),None,jnp.ones(1))
    tokens,valid,ar,_=Pi0.embed_suffix(fake,obs,jnp.zeros((1,30,48)),jnp.ones(1),condition)
    assert tokens.shape==(1,46,8)
    np.testing.assert_array_equal(tokens[:,:16],1)
    np.testing.assert_array_equal(tokens[:,16:],0)
    attention=np.asarray(make_attn_mask(valid,ar))[0]
    assert attention[16:,:16].all() and not attention[:16,16:].any()
