"""Explicit inference choices for the paper and EGG presets."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    profile: str = "paper"
    visual_offset: float = 0.0
    visual_scale: float = 1.0
    residual_bound: float = 0.1
    gate_kappa: float = 1.0
    maximum_tactile_decrement: float = 0.1
    forecast_horizon: int = 30
    tactile_fusion: str = "context_concat"
    seed: int = 43

    def __post_init__(self):
        numbers = (self.visual_offset, self.visual_scale, self.residual_bound,
                   self.gate_kappa, self.maximum_tactile_decrement)
        if not all(type(v) in (int, float) and math.isfinite(v) for v in numbers):
            raise ValueError("Inference parameters must be finite numbers")
        if not (0 <= self.visual_offset < 1 and 0 < self.visual_scale <= 1
                and self.visual_offset + self.visual_scale <= 1):
            raise ValueError("Visual mapping must map [0,1] into [0,1]")
        if not (0 <= self.residual_bound <= .1 and 0 < self.gate_kappa <= 10
                and 0 <= self.maximum_tactile_decrement <= .1):
            raise ValueError("Invalid residual bound, kappa or tactile decrement")
        if type(self.forecast_horizon) is not int or not 1 <= self.forecast_horizon <= 30:
            raise ValueError("forecast_horizon must be an integer in [1,30]")
        if self.tactile_fusion not in ("context_concat", "action_adapter"):
            raise ValueError("Unknown tactile fusion")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if self.profile not in ("paper", "egg", "custom"):
            raise ValueError("Unknown profile")
        if self.profile == "paper" and (self.visual_offset != 0 or self.visual_scale != 1
                or self.residual_bound != .1 or self.forecast_horizon != 30
                or self.tactile_fusion != "context_concat"):
            raise ValueError("Paper profile requires identity visual score, b=.1, H=30 and context concatenation")

    @classmethod
    def preset(cls, name="paper"):
        if name == "paper":
            return cls()
        if name == "egg":
            return cls(profile="egg", visual_offset=.6, visual_scale=.4,
                       residual_bound=.01, forecast_horizon=5, tactile_fusion="action_adapter")
        raise ValueError("Preset must be paper or egg")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown configuration fields")
        return cls(**data)

    @classmethod
    def load(cls, name_or_path="paper"):
        if str(name_or_path) in ("paper", "egg"):
            return cls.preset(str(name_or_path))
        return cls.from_dict(json.loads(Path(name_or_path).read_text()))

    def as_dict(self):
        return asdict(self)
