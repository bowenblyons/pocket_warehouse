from pocket_warehouse.triage.margin import calculate_cost


def test_replace_not_stocked(test_config):
    cost = calculate_cost("wheel", "replace", False, test_config)
    assert cost == 1.25


def test_replace_stocked(test_config):
    cost = calculate_cost("wheel", "replace", True, test_config)
    assert cost == 1.00


def test_repair_cost(test_config):
    cost = calculate_cost("wheel", "repair", True, test_config)
    assert cost == 1.25
