from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from tipdex.schedule import Plan, ActionSchedule
from tipdex.runtime import ClosedLoop


def test_replacement_never_moves_time_or_committed_slots():
    plan=Plan('s',0,0,np.arange(30)/30.,np.zeros((30,13)))
    schedule=ActionSchedule(plan);schedule.commit([8,15])
    mask=schedule.replace(np.ones((30,13)),.15,np.ones(30,bool))
    assert not mask[:5].any() and not mask[8] and not mask[15]
    assert mask[9] and np.all(schedule.plan.commands[8]==0)
    np.testing.assert_array_equal(schedule.plan.times,plan.times)


def fake_loop():
    policy=SimpleNamespace(session_id='s',fps=30.,observation_time=100.)
    policy.fast=Mock(return_value=dict(observation_time=100.,fps=30.,valid_from=2,
        actions=np.ones((30,13)).tolist(),q_v=.8,q_exec=.7))
    loop=ClosedLoop(policy,clock=lambda:100.05);loop.chunk=1
    loop.issue_forecast=Mock(return_value={'forecast_id':'new'})
    return loop


def test_accepted_update_triggers_new_forecast_and_preserves_protection():
    loop=fake_loop();result=loop.fast_update(100.04,np.zeros(19))
    loop.issue_forecast.assert_called_once()
    assert result['forecast']['forecast_id']=='new'
    loop.commit([6,9]);before=loop.schedule.plan.commands.copy()
    loop.policy.fast.return_value['actions']=(np.ones((30,13))*2).tolist()
    result=loop.fast_update(100.045,np.zeros(19))
    np.testing.assert_array_equal(np.asarray(result['actions'])[[6,9]],before[[6,9]])
    passed=loop.policy.fast.call_args.kwargs['replaceable']
    assert not passed[6] and not passed[9] and loop.issue_forecast.call_count==2


def test_no_accepted_replacement_means_no_forecast():
    loop=fake_loop();r=loop.fast_update(100.04,np.zeros(19),allowed=np.zeros(30,bool))
    loop.issue_forecast.assert_not_called()
    assert r['forecast'] is None and not r['replaced']


def test_committed_intervals_survive_new_chunk():
    loop=fake_loop();loop.fast_update(100.04,np.zeros(19));loop.commit([8],chunk_id=1)
    loop.chunk=2;loop.schedule=None;loop.policy.observation_time=100.06;loop.clock=lambda:100.08
    loop.policy.fast.return_value['observation_time']=100.06
    result=loop.fast_update(100.07,np.zeros(19))
    assert 6 not in result['replaced'] and 7 not in result['replaced']
    assert 9 in result['replaced']


def test_invalid_and_wrong_chunk_commits_rejected():
    import pytest
    loop=fake_loop();loop.fast_update(100.04,np.zeros(19))
    for indices in ([-1],[30],[.5],[True]):
        with pytest.raises(ValueError):loop.commit(indices)
    with pytest.raises(ValueError):loop.commit([4],chunk_id=99)


def test_full_forecast_renewal_and_verification_timeline():
    import threading
    from tipdex.forecast.verifier import ForecastVerifier, NORMALIZATION
    from tipdex.config import InferenceConfig
    from tipdex.sensors import NativeF1Stream
    from tipdex.replay import check_closed_loop
    class Policy:
        config=InferenceConfig();fps=30.;session_id='s'
        sensor_lock=threading.RLock()
        stream=NativeF1Stream(max_age=.1)
        def __init__(self):
            self.forecast=SimpleNamespace(verifier=ForecastVerifier('test-host'),clock=lambda:100.,predict=self.predict)
            self.reset()
        def reset(self):
            self.stream.reset();self.forecast.verifier.reset();self.forecast.verifier.session='s';self.last_cutoff=-1
            return 's'
        def feed_tactile(self,sid,raw,valid,stamp):self.stream.update(raw,valid,stamp)
        def feed_frames(self,sid,frames):
            for f in frames:self.feed_tactile(sid,f['raw364'],f['valid'],f['timestamp'])
            self.forecast.verifier.observe(sid,frames)
        def acknowledge(self,sid,packets):return self.forecast.verifier.acknowledge(sid,packets)
        def slow(self,sid,images,state,stamp):self.observation_time=stamp;return dict(chunk_id=1)
        def fast(self,sid,chunk,stamp,*args,**kwargs):
            assert args[0] >= self.last_cutoff
            self.last_cutoff=int(np.ceil((stamp-100.)*30.))
            _,receipt=self.stream.snapshot(stamp)
            score=self.forecast.verifier.score(.8,dict(error_threshold_low=.005,error_threshold_full=.01),
                now=stamp,receipt_time=receipt)
            return dict(actions=np.ones((30,13)).tolist(),valid_from=2,observation_time=100.,fps=30.,**score)
        def predict(self,request):
            result=dict(forecast_id=str(request['plan_id']),session_id='s',plan_id=request['plan_id'],
                issued_at=self.forecast.clock(),target_times=request['target_times'],
                predicted_normalized=np.zeros((30,364)).tolist(),normalization_id=NORMALIZATION)
            self.forecast.verifier.register(request,result)
            return result
    fixture=dict(calibration_touch=np.zeros((16,364)),touch=np.zeros((16,364)),
        calibration_valid=np.ones(16,bool),frame_valid=np.ones(16,bool),state=np.zeros(19),images=np.zeros((3,224,224,3),np.uint8))
    result=check_closed_loop(Policy(),fixture)
    assert result['forecasts_issued']==2 and result['current_frame_verified']
