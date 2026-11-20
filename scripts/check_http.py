"""Exercise a local model service with archived observations and simulated SDK feedback.

No action output is sent to an actuator. Run this client on the same host as
its dedicated demonstration model service so both use one monotonic clock.
"""
import argparse
import json,time,threading,urllib.request,urllib.error
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:8116')
parser.add_argument('--example',required=True)
parser.add_argument('--output',default='outputs/http_check.json')
args=parser.parse_args()
root=args.url.rstrip('/')
def get(path):
    return urllib.request.urlopen(root+path,timeout=30).read()
def post(path,body):
    request=urllib.request.Request(root+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
        return json.load(urllib.request.urlopen(request,timeout=30))
    except urllib.error.HTTPError as e:
        raise RuntimeError(path+': '+e.read().decode()) from e
status=json.loads(get('/status'));assert status['ready']
assert 'TipDex_VTLA' in get('/config').decode()
assert 'TACE' in get('/dashboard').decode()
presets=json.loads(get('/presets'));assert presets['paper']['forecast_horizon']==30
assert post('/validate-config',presets['egg'])['visual_offset']==.6
sid=post('/reset',{})['session_id']
a=dict(np.load(args.example,allow_pickle=False))
state=a['state'].tolist();raw=a['touch'][-1].astype(int).tolist();command=state[:13]
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
stop=threading.Event();errors=[]
def feedback():
    seq=0
    while not stop.is_set():
        try:
            stamp=time.monotonic();seq+=1
            packets=[dict(kind='sdk_write_ack',schema=1,role=role,host_boot_id=boot,
                driver_boot_id='http-smoke-'+role,units='radians',success=True,sequence=seq,
                positions=part,ack_at=stamp,lost=0) for role,part in [('arm',command[:7]),('hand',command[7:])]]
            post('/feedback',dict(session_id=sid,frames=[dict(raw364=raw,valid=True,acquisition_time=stamp,timestamp=stamp)],
                states=[dict(timestamp=stamp,state=state)],acks=packets))
            stop.wait(.01)
        except Exception as e:errors.append(str(e));break
thread=threading.Thread(target=feedback,daemon=True);thread.start()
try:
    time.sleep(.8)
    # Use model-resolution RGB to avoid unrelated transport overhead.
    from openpi.shared.numpy_image_tools import resize_with_pad
    images=resize_with_pad(a['images'],224,224)
    from tipdex.wire import encode_rgb
    observation_stamp=time.monotonic()
    body=dict(session_id=sid,state=state,images={k:encode_rgb(images[i]) for i,k in enumerate(
        ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'])})
    body['timestamp']=observation_stamp
    slow=post('/slow',body)
    slow_age=time.monotonic()-body['timestamp']
    print('slow_timing',slow,'age',slow_age,flush=True)
    fast=post('/fast',dict(session_id=sid,timestamp=time.monotonic(),state=state,executed_steps=0))
    assert np.shape(fast['actions'])==(30,13) and fast['tace_active']
    assert fast['q_v'] is not None and fast['q_exec'] is not None
    assert fast['replaced']
    live=json.loads(get('/status'))
    assert live['last']['q_v']==fast['q_v']
    assert fast['forecast'] is not None,fast.get('forecast_unavailable_reason')
    report=dict(ready=True,config_http=True,dashboard_http=True,presets_validated=True,
        actual_model_slow=True,actual_model_fast=True,actual_forecast=True,
        forecast_horizon=len(fast['forecast']['target_times']),q_v=fast['q_v'],q=fast['q_exec'],
        q_visible_in_status=True,tace_active=fast['tace_active'],replaced=len(fast['replaced']),
        rgb_transport='lossless_png',slow_compute_ms=slow['slow_ms'],slow_roundtrip_age_ms=slow_age*1000,
        feedback_errors=errors,hardware_commands_sent=False,
        execution_evidence='synthetic HTTP feedback; model plans are not sent to actuators')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
finally:
    stop.set();thread.join(timeout=5)
    post('/reset',{})
