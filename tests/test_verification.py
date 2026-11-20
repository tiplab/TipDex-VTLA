"""Current-frame causal verification using the production SDK-evidence schema."""
import copy
import numpy as np
import pytest
from tipdex.forecast.verifier import ForecastVerifier, NORMALIZATION

CONTRACT={'error_threshold_low':.005,'error_threshold_full':.01}


def packet(role,t,sequence,value=0.):
    return dict(kind='sdk_write_ack',schema=1,role=role,host_boot_id='test-host',
        driver_boot_id=role+'-driver',units='radians',success=True,sequence=sequence,
        positions=[value]*(7 if role=='arm' else 6),ack_at=t,lost=0)


def setup():
    verifier=ForecastVerifier('test-host');verifier.session='session'
    for i,t in enumerate(np.arange(0.,.081,.01)):
        for role in ('arm','hand'):verifier.execution.ingest(packet(role,float(t),i+1),now=.1)
    request=dict(session_id='session',chunk_id=1,plan_id='session:1:1',anchor_time=.02,
        history_times=[0.,.01,.02],command_history=np.zeros((3,13)).tolist(),
        target_times=[.05,.08,.11,.14,.17],future_commands=np.zeros((5,13)).tolist())
    result=dict(session_id='session',plan_id=request['plan_id'],forecast_id='first',issued_at=.03,
        target_times=request['target_times'],predicted_normalized=np.zeros((5,364)).tolist(),
        normalization_id=NORMALIZATION)
    verifier.register(request,result)
    verifier.observe('session',[dict(acquisition_time=.05,timestamp=.052,raw364=[65535]*364,valid=True)],now=.06)
    return verifier,request,result


def test_first_frame_changes_score_without_waiting_for_five():
    v,_,_=setup();s=v.score(.8,CONTRACT,now=.055,receipt_time=.052)
    assert s['tactile_verification']=='verified' and s['forecast_frame_index']==0
    assert s['q_base']==pytest.approx(.8) and s['q_exec']==pytest.approx(.7)
    assert s['tactile_measurement_time']==.05 and s['tactile_receipt_time']==.052


def test_wrong_receipt_does_not_select_another_frame():
    v,_,_=setup();s=v.score(.8,CONTRACT,now=.055,receipt_time=.05)
    assert s['tactile_verification']=='unavailable' and s['tactile_error'] is None
    assert s['q_exec']==s['q_base']


def test_latest_eligible_prediction_and_immutable_registration():
    v,r,p=setup();p['forecast_id']='new';p['issued_at']=.04
    p['predicted_normalized']=np.ones((5,364)).tolist();v.register(r,p)
    p['predicted_normalized'][0]=[0]*364;r['future_commands'][0]=[1]*13
    s=v.score(.8,CONTRACT,now=.055)
    assert s['forecast_id']=='new' and s['tactile_error']==0


def test_wrong_commands_and_missing_acks_fall_back():
    v,r,p=setup();v.execution.events['arm'][4]=(.04,np.ones(7))
    assert v.score(.8,CONTRACT,now=.055)['tactile_verification']=='unavailable'
    v,_,_=setup();v.execution.events['hand'].clear()
    assert v.score(.8,CONTRACT,now=.055)['tactile_verification']=='unavailable'


def test_future_condition_changes_do_not_block_current_frame():
    v,r,p=setup();v.pending.clear();r['future_commands'][1:]=np.ones((4,13)).tolist()
    v.register(r,p)
    assert v.score(.8,CONTRACT,now=.055)['tactile_verification']=='verified'


def test_future_samples_or_stale_measurements_not_used():
    v,_,_=setup()
    assert v.score(.8,CONTRACT,now=.051)['tactile_verification']=='unavailable'
    assert v.score(.8,CONTRACT,now=.2)['tactile_verification']=='unavailable'


@pytest.mark.parametrize('change', ['retro','nan','normalization','session','plan','commands'])
def test_malformed_forecast_rejected(change):
    v,r,p=setup()
    if change=='retro':p['issued_at']=.05
    elif change=='nan':p['predicted_normalized'][0][0]=float('nan')
    elif change=='normalization':p['normalization_id']='other'
    elif change=='session':r['session_id']='old'
    elif change=='plan':p['plan_id']='another'
    elif change=='commands':r['future_commands']=[[0.]*12]*5
    with pytest.raises(ValueError):v.register(r,p)


def test_timeline_is_not_retimed_to_tactile_targets():
    v,r,p=setup();r['command_times']=[.02,.045,.075]
    r['scheduled_commands']=np.zeros((3,13)).tolist()
    v.register(r,p)
    assert v.score(.8,CONTRACT,now=.055)['tactile_verification']=='verified'
    r['scheduled_commands'][1]=[1]*13
    with pytest.raises(ValueError,match='samples differ'):v.register(r,p)


def test_restarted_driver_never_verifies():
    v,_,_=setup();event=packet('arm',.09,10);event['driver_boot_id']='new'
    v.execution.ingest(event,now=.1)
    assert v.score(.8,CONTRACT,now=.055)['tactile_verification']=='unavailable'
