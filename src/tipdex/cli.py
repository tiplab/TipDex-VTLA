"""Command line entry points for bundles, replay and serving."""
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description='TipDex_VTLA official inference example')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('verify','replay','serve'):
        p=sub.add_parser(name)
        p.add_argument('--model',required=True,help='Local bundle or Hugging Face repository ID')
        p.add_argument('--revision',help='Hugging Face commit SHA')
        p.add_argument('--offline',action='store_true')
        p.add_argument('--config',default='paper',help='paper, tipdex or configuration JSON')
        if name=='replay':
            p.add_argument('--output',default='outputs/replay')
            p.add_argument('--stress-error',action='store_true',help='Inject a large error to test the TACE path')
        if name=='serve':
            p.add_argument('--host',default='127.0.0.1')
            p.add_argument('--port',type=int,default=8116)
    p=sub.add_parser('config');p.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    if args.command=='config':
        from .server import serve_config
        return serve_config(args.port)
    if args.command=='verify':
        from .bundle import ModelBundle
        bundle=ModelBundle.load(args.model,revision=args.revision,local_files_only=args.offline)
        print(json.dumps(dict(verified=True,checkpoint_step=9309,files=len(bundle.manifest['files']))))
        return
    from .policy import TipDexPolicy
    policy=TipDexPolicy(args.model,config=args.config,revision=args.revision,local_files_only=args.offline)
    from .replay import run_replay
    if args.command=='replay':
        print(json.dumps(run_replay(policy,output=args.output,stress_error=args.stress_error)))
    else:
        from .server import serve_policy
        serve_policy(policy,args.host,args.port)

if __name__=='__main__':main()
