import json
import threading
from unittest.mock import Mock
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
import pytest
from tipdex.bundle import ModelBundle, SCHEMA, sha256, relative_path
from tipdex.config import InferenceConfig
from tipdex.server import make_handler


def small_bundle(root):
    required=('backbone/config.json','backbone/params/_METADATA',
        'backbone/assets/tipdex/norm_stats.json','tokenizer/paligemma_tokenizer.model',
        'VASE/metadata.json','VASE/weights.npz','TACE/tace.msgpack',
        'forecast/contract.json','forecast/normalization.npz','forecast/weights.pt')
    for name in required:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    relationship={key:sha256(root/name) for key,name in {
        'weights_sha256':'TACE/tace.msgpack','vase_weights_sha256':'VASE/weights.npz',
        'backbone_metadata_sha256':'backbone/params/_METADATA',
        'normalization_sha256':'backbone/assets/tipdex/norm_stats.json'}.items()}
    (root/'TACE/config.json').write_text(json.dumps(relationship))
    manifest=dict(schema=SCHEMA,checkpoint_step=9309,files={str(p.relative_to(root)):
        dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in root.rglob('*') if p.is_file()})
    (root/'manifest.json').write_text(json.dumps(manifest))


def test_bundle_checks_integrity_and_missing_files(tmp_path):
    small_bundle(tmp_path);ModelBundle.load(tmp_path)
    (tmp_path/'forecast/weights.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='integrity'):ModelBundle.load(tmp_path)


@pytest.mark.parametrize('directory', ['tipdex', 'previous_dataset'])
def test_bundle_resolves_manifest_paths_independent_of_dataset_name(tmp_path, directory):
    small_bundle(tmp_path)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    original = 'backbone/assets/tipdex/norm_stats.json'
    target = f'backbone/assets/{directory}/norm_stats.json'
    (tmp_path/target).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path/original).rename(tmp_path/target)
    manifest['files'][target] = manifest['files'].pop(original)
    sample = tmp_path/'example/observation.npz'
    sample.parent.mkdir();sample.write_bytes(b'sample')
    manifest['files']['example/observation.npz'] = dict(bytes=6, sha256=sha256(sample))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    bundle = ModelBundle.load(tmp_path)
    assert bundle.normalization_path == tmp_path/target
    assert bundle.example_path == sample


@pytest.mark.parametrize('count', [0, 2])
def test_bundle_rejects_missing_or_ambiguous_normalization(tmp_path, count):
    small_bundle(tmp_path)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    name = 'backbone/assets/tipdex/norm_stats.json'
    if count == 0:
        del manifest['files'][name]
    else:
        other = 'backbone/assets/another/norm_stats.json'
        (tmp_path/other).parent.mkdir(parents=True)
        (tmp_path/other).write_bytes((tmp_path/name).read_bytes())
        manifest['files'][other] = manifest['files'][name]
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Expected one bundle artifact'):
        ModelBundle.load(tmp_path)


def test_bundle_discovery_keeps_component_hash_validation(tmp_path):
    small_bundle(tmp_path)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    name = 'backbone/assets/tipdex/norm_stats.json'
    (tmp_path/name).write_bytes(b'different normalization')
    manifest['files'][name] = dict(bytes=(tmp_path/name).stat().st_size, sha256=sha256(tmp_path/name))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Component compatibility mismatch: normalization_sha256'):
        ModelBundle.load(tmp_path)


@pytest.mark.parametrize('value',['/etc/passwd','../weights','a/../../x','a\\b','a//b',''])
def test_bundle_rejects_unsafe_paths(value):
    with pytest.raises(ValueError):relative_path(value)


def test_hf_loader_pins_revision_and_passes_offline_setting(tmp_path,monkeypatch):
    import huggingface_hub
    small_bundle(tmp_path)
    download=Mock(return_value=str(tmp_path));monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    ModelBundle.load('org/model',revision='a'*40,local_files_only=True)
    download.assert_called_once_with(repo_id='org/model',revision='a'*40,local_files_only=True)
    with pytest.raises(ValueError):ModelBundle.load('org/model',revision='main')


def test_config_http_presets_and_validation():
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler())
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    root=f'http://127.0.0.1:{server.server_port}'
    try:
        presets=json.load(urlopen(root+'/presets'))
        assert presets['paper']['forecast_horizon']==30
        assert presets['tipdex']['visual_offset']==.6
        assert 'TipDex_VTLA' in urlopen(root+'/config').read().decode()
        req=Request(root+'/validate-config',data=json.dumps(presets['tipdex']).encode(),headers={'Content-Type':'application/json'})
        assert json.load(urlopen(req))==InferenceConfig.preset('tipdex').as_dict()
        presets['paper']['forecast_horizon']=5
        req=Request(root+'/validate-config',data=json.dumps(presets['paper']).encode())
        with pytest.raises(HTTPError) as err:urlopen(req)
        assert err.value.code==400
    finally:
        server.shutdown();server.server_close();thread.join()


def test_model_requests_use_persistent_executor():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from tipdex.wire import encode_rgb
    import numpy as np
    calls=[]
    def slow(images,state,stamp):
        calls.append(threading.current_thread().name)
        return {'q_v':.8}
    loop=SimpleNamespace(session='s',slow_update=slow)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='tipdex-model-test') as executor:
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(loop,executor))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            body=dict(session_id='s',timestamp=1.,state=[0.]*19,
                      images={'base_0_rgb':encode_rgb(np.zeros((4,4,3),np.uint8))})
            for _ in range(2):
                req=Request(f'http://127.0.0.1:{server.server_port}/slow',data=json.dumps(body).encode())
                assert json.load(urlopen(req))['q_v']==.8
            assert calls[0]==calls[1] and calls[0].startswith('tipdex-model-test')
        finally:server.shutdown();server.server_close();thread.join()
