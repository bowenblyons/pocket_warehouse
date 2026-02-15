
INVENTORY = {
    "axle": 1,
    "wheel": 2,
    "body": 0,
    "frame": 0,
    "paint": int("inf")
}

def check_availability(part_name: str, quantity: int) -> bool:
    if part_name in INVENTORY:
        return quantity <= INVENTORY[part_name]
    return False
