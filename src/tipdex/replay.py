"""Real-weight offline replay with explicitly simulated command evidence."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def run_replay(policy, *, output=None, stress_error=False):
    from tipdex.policy import TipDexPolicy
    from tipdex.schema import IMAGE_KEYS
    import jax,jax.numpy as jnp
    import dataclasses
    p=policy
    config=p.config
    root=p.bundle.root
    weight_paths=[root/'VASE/weights.npz',root/'TACE/tace.msgpack',
                  p.forecast.directory/'weights.pt']
    before={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in weight_paths}
    a=dict(np.load(root/'example/egg.npz',allow_pickle=False))
    sid=p.reset()
    for name,valid,end in [('calibration_touch','calibration_valid',99.),('touch','frame_valid',100.)]:
        for j in range(16):p.feed_tactile(sid,a[name][j],bool(a[valid][j]),end+(j-15)/30.)
    images={k:a['images'][i] for i,k in enumerate(IMAGE_KEYS)}
    slow=p.slow(sid,images,a['state'],100.)
    shared_audit={}
    if True:
        from tipdex.forecast.shared_encoder import parameter_digest
        shared=p.forecast.model.encoder
        assert shared.encoder is p.model.tactile_encoder
        shared_before=parameter_digest(shared.encoder)
        touch_fields,_=p.stream.snapshot(100.)
        actual_features=shared.encode(touch_fields['tactile'][0,:,0,:],
            touch_fields['tactile_baseline'],touch_fields['tactile_sample_valid'])
        policy_features=np.asarray(p.model.encode_tactile_condition(p.observation)[0])[0]
        feature_error=float(np.max(abs(actual_features-policy_features)))
        assert feature_error<1e-5,feature_error
        shared_audit=dict(same_encoder_object=True,feature_max_abs_error=feature_error,
            encoder_parameters_sha256=shared_before)
    command=np.asarray(a['state'][:13],np.float64)
    hist=100.+np.arange(-15,1)/30.;targets=100.+np.arange(1,config.forecast_horizon+1)/30.
    request=dict(session_id=sid,chunk_id=slow['chunk_id'],plan_id=f'{sid}:{slow["chunk_id"]}:1',anchor_time=100.,
        history_times=hist.tolist(),touch_history=a['touch'].tolist(),
        state_history=np.tile(a['state'],(16,1)).tolist(),command_history=np.tile(command,(16,1)).tolist(),
        target_times=targets.tolist(),future_commands=np.tile(command,(config.forecast_horizon,1)).tolist(),
        command_times=[100.],scheduled_commands=[command.tolist()])
    p.forecast.clock=lambda:100.001
    forecast=p.forecast.predict(request)
    v=p.forecast.verifier
    for sequence,stamp in enumerate(np.arange(99.49,100.061,.01),1):
        for role,values in [('arm',command[:7]),('hand',command[7:])]:
            v.execution.ingest(dict(kind='sdk_write_ack',schema=1,role=role,host_boot_id=v.execution.host_boot_id,
                driver_boot_id='replay-'+role,units='radians',success=True,sequence=sequence,
                positions=values.tolist(),ack_at=float(stamp),lost=0),now=101.)
    # Explicit injected tactile discrepancy tests the large-error branch. No
    # robot is connected and this is not a forecast-accuracy measurement.
    actual=np.full(364,65535,dtype=np.uint16) if stress_error else a['touch'][-1].astype(np.uint16)
    receipt=float(targets[0]+.001)
    p.feed_tactile(sid,actual,True,receipt)
    v.observe(sid,[dict(acquisition_time=float(targets[0]),timestamp=receipt,raw364=actual.tolist(),valid=True)],now=100.06)
    fast=p.fast(sid,slow['chunk_id'],100.04,feedback_state=a['state'])
    assert fast['tactile_verification']=='verified', fast
    assert fast['forecast_frame_index']==0
    if stress_error: assert fast['q_exec'] < fast['q_base']
    assert fast['tace_active'] is True
    fields,_=p.stream.snapshot(100.04)
    obs=dataclasses.replace(p.observation,**{k:jnp.asarray(val)[None] for k,val in fields.items()})
    progress=jnp.asarray([fast['valid_from']],jnp.int32)
    state=jnp.asarray(p.tace.state)[None]
    low=p.tace.delta(p.model,p.tace.params,p.snapshot,obs,state,jnp.asarray([fast['q_exec']],jnp.float32),progress,jnp.ones((1,30),bool))
    high=p.tace.delta(p.model,p.tace.params,p.snapshot,obs,state,jnp.asarray([fast['q_base']],jnp.float32),progress,jnp.ones((1,30),bool))
    low,high=np.asarray(low),np.asarray(high)
    assert np.isfinite(low).all() and np.isfinite(high).all()
    protected=float(np.max(np.abs(low[:,:fast['valid_from']])))
    effect=float(np.max(np.abs(low-high)))
    assert protected==0.
    if stress_error: assert effect>0.,effect
    after={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in weight_paths}
    assert before==after
    if shared_audit:
        assert parameter_digest(p.model.tactile_encoder)==shared_before
        shared_audit['encoder_parameters_unchanged']=True
    report=dict(model='9309',profile=config.profile,actual_weights_sha256=before,weights_unchanged=True,
        forecast_shape=list(np.shape(forecast['predicted_normalized'])),
        q_v=fast['q_v'],q_base=fast['q_base'],q_exec=fast['q_exec'],verified_error=fast['tactile_error'],
        verification=fast['tactile_verification'],frame_index=fast['forecast_frame_index'],
        error_aggregation=fast['tactile_error_aggregation'],tace_active=fast['tace_active'],
        max_normalized_action_change_due_to_error=effect,protected_action_delta=protected,
        model_action_shape=list(np.shape(fast['actions'])),all_actions_finite=bool(np.isfinite(fast['actions']).all()),
        shared_encoder=shared_audit,
        command_evidence='synthetic timestamped SDK acknowledgements for a held command',
        measurement=('raw364 stress injection' if stress_error else 'archived current tactile frame'), accuracy_evaluation=False,hardware_commands_sent=False)
    if stress_error:
        report["closed_loop"] = check_closed_loop(p,a)
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'prediction.npz',actions=np.asarray(fast['actions']),forecast=np.asarray(forecast['predicted_normalized']).reshape(config.forecast_horizon,14,26))
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    p.forecast.clock = time.monotonic
    return report


def check_closed_loop(policy, fixture):
    """Two real model updates with a deterministic simulated execution clock."""
    from .runtime import ClosedLoop
    from .schema import IMAGE_KEYS
    now=[100.05]
    loop=ClosedLoop(policy,clock=lambda:now[0]);session=loop.reset()
    policy.forecast.clock=lambda:now[0]
    for name,valid,end in [('calibration_touch','calibration_valid',99.),('touch','frame_valid',100.)]:
        for i in range(16):
            policy.feed_tactile(session,fixture[name][i],bool(fixture[valid][i]),end+(i-15)/30.)
    for stamp in np.arange(99.45,100.051,.01):loop.feed_state(round(float(stamp),9),fixture['state'])
    command=np.asarray(fixture['state'][:13],np.float64)
    evidence=policy.forecast.verifier.execution
    sequence=[0]
    def acknowledge(stamp, values):
        sequence[0]+=1
        packets=[]
        for role,part in [('arm',values[:7]),('hand',values[7:])]:
            packets.append(dict(kind='sdk_write_ack',schema=1,role=role,
                host_boot_id=evidence.host_boot_id,driver_boot_id='closed-loop-'+role,
                units='radians',success=True,sequence=sequence[0],positions=part.tolist(),ack_at=float(stamp),lost=0))
        loop.acknowledge(packets)
    for stamp in np.arange(99.45,100.051,.01):acknowledge(round(float(stamp),9),command)
    images={k:fixture['images'][i] for i,k in enumerate(IMAGE_KEYS)}
    loop.slow_update(images,fixture['state'],100.)
    first=loop.fast_update(100.04,fixture['state'])
    assert first['forecast'] is not None,first['forecast_unavailable_reason']
    target=first['forecast']['target_times'][0]
    plan=loop.schedule.plan
    # Replay exactly the accepted command transitions, with 100 Hz SDK ACKs.
    end=target+.003
    stamps=sorted(set(np.arange(100.06,end+.001,.01).tolist()+[end]+
                      [float(t) for t in plan.times if 100.05<t<=end]))
    for stamp in stamps:
        values=command if stamp<plan.times[first['replaced'][0]] else plan.at(stamp)
        acknowledge(stamp,values)
    now[0]=target+.01
    loop.feed_state(target+.001,fixture['state'])
    loop.feed_frames([dict(acquisition_time=target,timestamp=target+.001,raw364=[65535]*364,valid=True)])
    loop.commit([8]);protected=loop.schedule.plan.commands[8].copy()
    progress=int(np.searchsorted(plan.times,target,side='right'))
    second=loop.fast_update(target+.005,fixture['state'],executed_steps=progress)
    assert second['tactile_verification']=='verified',second.get('tactile_reason')
    assert second['q_exec']<second['q_base']
    assert second['forecast'] is not None,second['forecast_unavailable_reason']
    assert second['forecast']['forecast_id']!=first['forecast']['forecast_id']
    np.testing.assert_array_equal(second['actions'][8],protected)
    np.testing.assert_array_equal(second['action_times'],first['action_times'])
    return dict(accepted_updates=2,forecasts_issued=2,
                current_frame_verified=True,score_decreased=True,
                committed_slot_preserved=True,action_times_preserved=True,
                execution='simulated SDK acknowledgements',hardware_commands_sent=False)
