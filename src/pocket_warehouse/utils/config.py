from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PartConfig:
    """Per part configuration info."""

    cost: float
    repair_cost: float
    stock: int
    restock_at: int


@dataclass
class Config:
    """Configuration info for the system."""

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


def load_config(path: Path = Path("config/config.yaml")) -> Config:

    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return Config.from_dict(raw)
