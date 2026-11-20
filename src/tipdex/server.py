"""Local configuration page and JSON inference service; no robot SDK calls."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import threading
from .config import InferenceConfig


def make_handler(loop=None, executor=None):
    lock=threading.RLock()
    ingest_lock=threading.RLock()
    def model_call(fn,*args,**kwargs):
        return fn(*args,**kwargs) if executor is None else executor.submit(fn,*args,**kwargs).result()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def send(self,status,value,kind='application/json'):
            raw=(json.dumps(value,allow_nan=False) if kind=='application/json' else value).encode()
            self.send_response(status);self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

        def do_GET(self):
            if self.path in ('/','/config'):
                self.send(200,files('tipdex').joinpath('static/config.html').read_text(),'text/html')
            elif self.path=='/dashboard':
                self.send(200,files('tipdex').joinpath('static/dashboard.html').read_text(),'text/html')
            elif self.path=='/presets':
                self.send(200,{k:InferenceConfig.preset(k).as_dict() for k in ('paper','egg')})
            elif self.path=='/status' and loop is not None:
                # Snapshot telemetry only; does not wait for model inference.
                self.send(200,{**loop.policy.status(),'last':loop.last})
            else:self.send(404,{'error':'unknown route'})

        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 32*1024*1024:
                    raise ValueError('Invalid JSON body size')
                body=json.loads(self.rfile.read(length))
                if self.path=='/validate-config':
                    return self.send(200,InferenceConfig.from_dict(body).as_dict())
                if loop is None:
                    return self.send(404,{'error':'no model loaded'})
                if self.path=='/reset':
                    with lock, ingest_lock:
                        return self.send(200,{'session_id':model_call(loop.reset)})
                if body.get('session_id')!=loop.session:
                    raise ValueError('Stale session')
                if self.path=='/feedback':
                    with ingest_lock:
                        if body.get('session_id')!=loop.session:raise ValueError('Stale session')
                        if body.get('frames'):loop.feed_frames(body['frames'])
                        for row in body.get('states',[]):loop.feed_state(row['timestamp'],row['state'])
                        result=loop.acknowledge(body['acks']) if body.get('acks') else {'accepted':True}
                elif self.path=='/commit':
                    with ingest_lock:
                        if body.get('session_id')!=loop.session:raise ValueError('Stale session')
                        loop.commit(body['indices'],chunk_id=body['chunk_id']);result={'accepted':True}
                elif self.path=='/slow':
                    with lock:
                        if body.get('session_id')!=loop.session:raise ValueError('Stale session')
                        from .wire import decode_rgb
                        images={k:decode_rgb(v) for k,v in body['images'].items()}
                        result=model_call(loop.slow_update,images,body['state'],body['timestamp'])
                elif self.path=='/fast':
                    with lock:
                        if body.get('session_id')!=loop.session:raise ValueError('Stale session')
                        result=model_call(loop.fast_update,body['timestamp'],body['state'],
                            executed_steps=body.get('executed_steps',0),allowed=body.get('replaceable'))
                else:return self.send(404,{'error':'unknown route'})
                self.send(200,result)
            except (KeyError,TypeError,ValueError) as error:
                self.send(400,{'error':str(error)})
    return Handler


def serve_config(port):
    print(f'Configuration: http://127.0.0.1:{port}/config',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),make_handler()).serve_forever()


def serve_policy(policy,host,port):
    from .runtime import ClosedLoop
    from concurrent.futures import ThreadPoolExecutor
    from .replay import run_replay
    def initialize():
        import torch
        torch.set_num_threads(1)
        # Warm the same thread that handles all GPU/forecast requests.
        run_replay(policy)
        policy.reset()
        return ClosedLoop(policy)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='tipdex-model') as executor:
        loop=executor.submit(initialize).result()
        print(f'TipDex_VTLA ready: http://{host}:{port}/dashboard',flush=True)
        server=ThreadingHTTPServer((host,port),make_handler(loop,executor))
        try:server.serve_forever()
        finally:server.server_close()
