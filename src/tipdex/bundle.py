"""Model bundles resolved locally or from a pinned Hugging Face revision."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

SCHEMA = "tipdex.inference_bundle.v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("Bundle paths must be POSIX relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != value:
        raise ValueError("Unsafe bundle path")
    return path


def unique_artifact(files, pattern):
    """Resolve a manifest artifact without depending on a dataset directory name."""
    matches = [name for name in files if relative_path(name).match(pattern)]
    if len(matches) != 1:
        raise ValueError(f"Expected one bundle artifact matching {pattern}; found {len(matches)}")
    return matches[0]


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    manifest: dict

    @classmethod
    def load(cls, source, *, revision=None, local_files_only=False):
        root = Path(source).expanduser()
        if not root.is_dir():
            if revision is None or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("Hugging Face loading requires a 40-character commit revision")
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(repo_id=str(source), revision=revision,
                                          local_files_only=local_files_only))
        manifest = json.loads((root/"manifest.json").read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("checkpoint_step") != 9309:
            raise ValueError("Unsupported TipDex model bundle")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Bundle has no integrity manifest")
        for name, item in files.items():
            path = root/relative_path(name)
            if (not isinstance(item, dict) or not path.is_file()
                    or path.stat().st_size != item.get("bytes") or sha256(path) != item.get("sha256")):
                raise ValueError(f"Bundle integrity failure: {name}")
        normalization = unique_artifact(files, "backbone/assets/*/norm_stats.json")
        required = ("backbone/config.json", "backbone/params/_METADATA",
                    "tokenizer/paligemma_tokenizer.model",
                    "VASE/metadata.json", "VASE/weights.npz", "TACE/config.json", "TACE/tace.msgpack",
                    "forecast/contract.json", "forecast/normalization.npz", "forecast/weights.pt")
        if not set(required) <= files.keys():
            raise ValueError("Incomplete model bundle")
        relationships = json.loads((root/"TACE/config.json").read_text())
        expected = {
            "weights_sha256": "TACE/tace.msgpack",
            "vase_weights_sha256": "VASE/weights.npz",
            "backbone_metadata_sha256": "backbone/params/_METADATA",
            "normalization_sha256": normalization,
        }
        for key, name in expected.items():
            if relationships.get(key) != files[name]["sha256"]:
                raise ValueError(f"Component compatibility mismatch: {key}")
        return cls(root.resolve(), manifest)

    def path(self, name):
        return self.root/relative_path(name)

    @property
    def normalization_path(self):
        return self.path(unique_artifact(self.manifest["files"], "backbone/assets/*/norm_stats.json"))

    @property
    def example_path(self):
        return self.path(unique_artifact(self.manifest["files"], "example/*.npz"))
