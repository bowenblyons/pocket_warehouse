from pocket_warehouse.utils.config import get_config
from pocket_warehouse.schemas.schemas import PartRequirement, InventoryImpact

def check_availability(part_name: str, quantity: int) -> bool:

    cfg = get_config()

    if part_name in cfg.parts:
        return quantity <= cfg.parts[part_name].stock
    return False

def get_impact(req: list[PartRequirement]) -> InventoryImpact|None:
    
    needed: dict[str, int] = {}
    remaining: dict[str, int] = {}
    reorders: list[str] = []
    
    if len(req) == 0:
        return None
    
    cfg = get_config()

    for part in req:
        needed[part.name] = 1
        remaining[part.name] = cfg.parts[part.name].stock - 1
        if remaining[part.name] <= cfg.parts[part.name].restock_at:
            reorders.append(part.name)
    
    return InventoryImpact(needed, remaining, reorders)