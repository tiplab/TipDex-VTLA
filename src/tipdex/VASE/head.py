"""Exact paper VASE labels and a strict, NumPy-only exported head runtime.

q_v estimates frozen-policy action consistency under illumination changes. It
is neither task success probability nor action correctness. This module never
relabels the output of an older floored or group-weighted/logistic head.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA = "tipdex.paper_vase.v1"
POOLING = "masked_mean_final_prefix_context"
PREDICTION_FORMULA = "q_v = sigmoid(h_omega(Pool(c_t)))"
TARGET_FORMULA = "q_v_target = exp(-paired_mse / lambda_scale)"
INPUT_DIM = 2048
HIDDEN_DIM = 128
LAYER_NORM_EPS = 1e-5
WEIGHT_SHAPES = {
    "ln_weight": (INPUT_DIM,), "ln_bias": (INPUT_DIM,),
    "fc1_weight": (HIDDEN_DIM, INPUT_DIM), "fc1_bias": (HIDDEN_DIM,),
    "fc2_weight": (1, HIDDEN_DIM), "fc2_bias": (1,),
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_paired_action_mse(clean_actions, perturbed_actions, valid) -> np.ndarray:
    """Per-example unweighted MSE, then equal mean over paired noise seeds.

    Inputs are [N,S,H,D], with an identical binary validity mask [N,H]. Action
    coordinates must already use the same frozen-policy normalization. Every
    valid scalar action coordinate has equal weight; no region weighting, square
    root, logistic transform, expert comparison, or sample-count weighting.
    """
    clean = np.asarray(clean_actions, dtype=np.float64)
    perturbed = np.asarray(perturbed_actions, dtype=np.float64)
    mask = np.asarray(valid)
    if clean.ndim != 4 or clean.shape != perturbed.shape or not all(clean.shape):
        raise ValueError("paired actions must have identical nonempty [N,S,H,D] shapes")
    if mask.shape != (clean.shape[0], clean.shape[2]):
        raise ValueError("valid must have shape [N,H]")
    if not np.isin(mask, [False, True]).all():
        raise ValueError("valid must be binary")
    if not np.isfinite(clean).all() or not np.isfinite(perturbed).all():
        raise ValueError("paired action inputs must be finite")
    mask = mask.astype(bool)
    if not mask.any(axis=1).all():
        raise ValueError("an example has no valid action coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.where(mask[:, None, :, None], perturbed - clean, 0.0)
        per_seed = np.square(difference).sum(axis=(2, 3)) / (
            mask.sum(axis=1)[:, None] * clean.shape[3])
        result = per_seed.mean(axis=1)
    if not np.isfinite(result).all():
        raise ValueError("paired MSE overflowed")
    return result


def paper_vase_targets(paired_mse, *, lambda_scale: float) -> np.ndarray:
    """Return exp(-paired_mse/lambda), without a floor or a fitted remapping."""
    if isinstance(lambda_scale, bool):
        raise ValueError("lambda_scale must be explicit, finite and positive")
    scale = float(lambda_scale)
    mse = np.asarray(paired_mse, dtype=np.float64)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("lambda_scale must be explicit, finite and positive")
    if not np.isfinite(mse).all() or np.any(mse < 0):
        raise ValueError("paired_mse must be finite and nonnegative")
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-mse / scale)


def raw_sigmoid(logits) -> np.ndarray:
    """Stable float32 sigmoid; no floor and no endpoint clipping."""
    logits = np.asarray(logits, dtype=np.float32)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _exact_gelu(values: np.ndarray) -> np.ndarray:
    # Matches torch.nn.GELU(approximate="none"), without a SciPy dependency.
    erf = np.frompyfunc(math.erf, 1, 1)(values / np.float32(math.sqrt(2.0)))
    return np.float32(0.5) * values * (np.float32(1.0) + erf.astype(np.float32))


def validate_metadata(metadata: Mapping) -> None:
    """Reject old/underspecified heads instead of converting their predictions."""
    required = {
        "schema": SCHEMA,
        "kind": "paper_visual_action_reliability",
        "prediction_formula": PREDICTION_FORMULA,
        "target_formula": TARGET_FORMULA,
        "pooling": POOLING,
    }
    for key, expected in required.items():
        if key not in metadata or metadata[key] != expected:
            raise ValueError(f"incompatible VASE metadata: {key}")
    floor = metadata.get("q_v_floor")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or floor != 0:
        raise ValueError("paper VASE requires explicit q_v_floor=0")
    architecture = metadata.get("architecture")
    expected_architecture = dict(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                                 output_dim=1, layer_norm_eps=LAYER_NORM_EPS,
                                 activation="GELU", gelu_approximate="none")
    if architecture != expected_architecture:
        raise ValueError("incompatible VASE architecture")
    paper_vase_targets([0], lambda_scale=metadata.get("lambda_scale", float("nan")))


class NumpyVASEHead:
    def __init__(self, weights: Mapping[str, np.ndarray], metadata: Mapping):
        validate_metadata(metadata)
        if set(weights) != set(WEIGHT_SHAPES):
            raise ValueError("unexpected or missing VASE weight arrays")
        self.weights = {}
        for name, shape in WEIGHT_SHAPES.items():
            value = np.array(weights[name], dtype=np.float32, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid VASE weights: {name}")
            value.setflags(write=False)
            self.weights[name] = value
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))

    @classmethod
    def load(cls, directory: Path | str) -> "NumpyVASEHead":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        validate_metadata(metadata)
        path = directory / "weights.npz"
        if metadata.get("weights_sha256") != sha256_file(path):
            raise ValueError("VASE weight hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            weights = {name: archive[name] for name in archive.files}
        return cls(weights, metadata)

    def logits(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != INPUT_DIM or not x.shape[0] or not np.isfinite(x).all():
            raise ValueError("features must be finite nonempty [N,2048] pooled prefix features")
        weights = self.weights
        with np.errstate(over="ignore", invalid="ignore"):
            centered = x - x.mean(axis=-1, keepdims=True)
            variance = np.square(centered).mean(axis=-1, keepdims=True)
            x = centered / np.sqrt(variance + np.float32(LAYER_NORM_EPS))
            x = x * weights["ln_weight"] + weights["ln_bias"]
            x = _exact_gelu(x @ weights["fc1_weight"].T + weights["fc1_bias"])
            result = (x @ weights["fc2_weight"].T + weights["fc2_bias"])[:, 0]
        if not np.isfinite(result).all():
            raise ValueError("nonfinite VASE inference result")
        return result

    def predict(self, features) -> np.ndarray:
        return raw_sigmoid(self.logits(features))
