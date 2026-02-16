from pocket_warehouse.schemas.schemas import InventoryImpact, PartRequirement
from pocket_warehouse.triage.inventory import check_availability, get_impact


def test_part_available(test_config):
    availability = check_availability("wheel", 1, test_config)
    assert availability


def test_part_not_available(test_config):
    availability = check_availability("axle", 1, test_config)
    assert availability is False


def test_part_needed_in_stock_with_reorder(test_config):
    part_req = get_impact(
        [PartRequirement("wheel", "replace", 0.0, False)], test_config
    )
    expected = InventoryImpact({"wheel": 1}, {"wheel": 3}, ["wheel"])
    assert part_req == expected


def test_part_needed_in_stock_no_reorder(test_config):
    part_req = get_impact(
        [PartRequirement("frame", "replace", 0.0, False)], test_config
    )
    expected = InventoryImpact({"frame": 1}, {"frame": 1}, [])
    assert part_req == expected


def test_part_needed_not_in_stock_with_reorder(test_config):
    part_req = get_impact(
        [PartRequirement("axle", "replace", 0.0, False)], test_config
    )
    expected = InventoryImpact({"axle": 1}, {"axle": 0}, ["axle"])
    assert part_req == expected
