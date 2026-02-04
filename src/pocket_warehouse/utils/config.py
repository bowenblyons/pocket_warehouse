from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Any

@dataclass
class PartConfig:
    cost: float
    repair_cost: float
    stock: float
    restock_at: float

@dataclass
class Config:
    resell_market_value: float
    labor_per_repair: float
    axle: PartConfig
    wheel: PartConfig
    frame: PartConfig
    body: PartConfig
    paint: PartConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        parts = {k: PartConfig(**v) for k, v in data.get("parts", {}).items()}
        return cls(
            resell_market_value=data["resell_market_value"],
            labor_per_repair=data["labor_per_repair"],
            **parts
        )

_config: Config | None = None

def load_config(path: Path = Path("config/config.yaml")) -> None:
    global _config
    if _config is None:
        with path.open("r") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
            
        _config = Config.from_dict(raw)

def get_config() -> Config:
    if _config is None:
        load_config()
    assert _config is not None
    return _config

load_config()
