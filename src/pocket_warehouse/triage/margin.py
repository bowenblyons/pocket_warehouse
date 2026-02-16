from pocket_warehouse.utils.config import Config


def calculate_cost(
    name: str, action: str, is_stocked: bool, cfg: Config
) -> float:
    """Calculates the total cost of the repair or replacement including labor"""

    cost: float = 0.0

    if action == "replace":
        if not is_stocked:
            cost += cfg.parts[name].cost
        cost += cfg.labor_per_repair

    elif action == "repair":
        cost += cfg.parts[name].repair_cost
        cost += cfg.labor_per_repair

    return cost
