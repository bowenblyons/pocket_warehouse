from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Any


@dataclass
class PartConfig:
    cost: float
    repair_cost: float
    stock: int
    restock_at: int


@dataclass
class Config:
    resell_market_value: float
    labor_per_repair: float
    confidence_threshold: float
    parts: dict[str, PartConfig]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":

        parts = {
            name: PartConfig(**values)
            for name, values in data.get("parts", {}).items()
        }

        return cls(
            resell_market_value=data["resell_market_value"],
            labor_per_repair=data["labor_per_repair"],
            parts=parts,
            confidence_threshold=data["confidence_threshold"],
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
