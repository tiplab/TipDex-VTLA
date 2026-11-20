"""Package existing checkpoints for the public inference loader; never upload."""
import argparse
from importlib.resources import files
import json
from pathlib import Path
import shutil
from .bundle import SCHEMA, sha256, ModelBundle


def export_bundle(checkpoint, vase, tace, forecast, tokenizer, example, output):
    sources = [Path(p).resolve() for p in (checkpoint,vase,tace,forecast,tokenizer,example)]
    checkpoint,vase,tace,forecast,tokenizer,example = sources
    root = Path(output).resolve()
    if root.exists():
        raise ValueError('Output already exists; use a new directory')
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
    meta = json.loads((vase/'metadata.json').read_text())
    protocol = json.loads((tace/'protocol.json').read_text())
    result = json.loads((tace/'result.json').read_text())
    native = json.loads((tace/'runtime_manifest.json').read_text())
    contract = json.loads((forecast/'contract.json').read_text())
    if (sha256(vase/'weights.npz') != meta['weights_sha256']
            or sha256(tace/'tace.msgpack') != result['weights_sha256']
            or protocol['vase_weight_hash'] != meta['weights_sha256']
            or sha256(checkpoint/'params/_METADATA') != native['backbone_metadata_sha256']
            or sha256(checkpoint/'assets/demo01_egg/norm_stats.json') != native['normalization_sha256']):
        raise ValueError('Backbone, VASE and TACE provenance mismatch')
    for name in ('weights.pt','normalization.npz'):
        if sha256(forecast/name) != contract['sha256'][name]:
            raise ValueError('Forecast artifact integrity mismatch')
    root.mkdir(parents=True)
    def copy(src, dst):
        target = root/dst
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target)
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    shutil.copytree(checkpoint/'params',root/'backbone/params')
    copy(checkpoint/'assets/demo01_egg/norm_stats.json','backbone/assets/demo01_egg/norm_stats.json')
    copy(checkpoint/'assets/fastslow.json','backbone/assets/fastslow.json')
    write('backbone/config.json',json.loads(files('tipdex').joinpath('static/backbone_9309.json').read_text()))
    copy(tokenizer,'tokenizer/paligemma_tokenizer.model')
    copy(example,'example/egg.npz')
    copy(vase/'weights.npz','VASE/weights.npz')
    # Runtime metadata contains only tensor contracts and scores, no private paths.
    keys=('schema','kind','prediction_formula','target_formula','pooling','q_v_floor',
          'architecture','lambda_scale','weights_sha256','score_meaning')
    write('VASE/metadata.json',{key:meta[key] for key in keys if key in meta})
    copy(tace/'tace.msgpack','TACE/tace.msgpack')
    write('TACE/config.json',dict(architecture='shared_gelu_feature_residual_v1',
        template_dims=result['template_dims'],weights_sha256=result['weights_sha256'],
        vase_weights_sha256=meta['weights_sha256'],
        backbone_metadata_sha256=native['backbone_metadata_sha256'],
        normalization_sha256=native['normalization_sha256']))
    for name in ('weights.pt','normalization.npz'): copy(forecast/name,'forecast/'+name)
    keys=('shared_encoder_parameters_sha256','sha256','error_threshold_low','error_threshold_full',
          'alignment_tolerance_seconds','measurement_max_age_seconds')
    portable={key:contract[key] for key in keys if key in contract}
    portable.update(architecture='shared_9309_recurrent_forecaster_v1', normalization='raw_uint16_divided_by_65535')
    write('forecast/contract.json',portable)
    for source, target in [('model_card.md','README.md'),('LICENSE.txt','LICENSE'),('LICENSE_GEMMA.txt','LICENSE_GEMMA.txt')]:
        (root/target).write_text(files('tipdex').joinpath('static/'+source).read_text())
    entries={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha256(p))
             for p in sorted(root.rglob('*')) if p.is_file()}
    write('manifest.json',dict(schema=SCHEMA,checkpoint_step=9309,
        model='TipDex_VTLA',files=entries,default_profile='paper'))
    ModelBundle.load(root)
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','vase','tace','forecast','tokenizer','example','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(export_bundle(**vars(args)))

if __name__=='__main__':main()
