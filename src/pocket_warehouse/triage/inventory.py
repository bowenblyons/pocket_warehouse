from pocket_warehouse.schemas.schemas import InventoryImpact, PartRequirement
from pocket_warehouse.utils.config import Config


def check_availability(part_name: str, quantity: int, cfg: Config) -> bool:
    """Checks availability of the part using the config info."""
    if part_name in cfg.parts:
        return quantity <= cfg.parts[part_name].stock
    return False


def get_impact(
    req: list[PartRequirement], cfg: Config
) -> InventoryImpact | None:
    """Calculates the impact to inventory and if a reorder is triggered."""

    needed: dict[str, int] = {}
    remaining: dict[str, int] = {}
    reorders: list[str] = []

    if len(req) == 0:
        return None

    for part in req:
        if part.action == "replace":
            needed[part.name] = 1

            if cfg.parts[part.name].stock >= 1:
                remaining[part.name] = cfg.parts[part.name].stock - 1
            else:
                remaining[part.name] = 0
                reorders.append(part.name)
            if remaining[part.name] <= cfg.parts[part.name].restock_at:
                reorders.append(part.name)

    return InventoryImpact(needed, remaining, reorders)
