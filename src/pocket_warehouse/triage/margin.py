from pocket_warehouse.utils.config import get_config


def calculate_cost(name: str, action: str, is_stocked: bool) -> float:

    cost: float = 0.0
    cfg = get_config()

    if action == "replace":
        if not is_stocked:
            cost += cfg.parts[name].cost
        cost += cfg.labor_per_repair

    elif action == "repair":
        cost += cfg.parts[name].repair_cost
        cost += cfg.labor_per_repair

    return cost
