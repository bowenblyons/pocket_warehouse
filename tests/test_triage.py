from pocket_warehouse.schemas.schemas import (
    ClassificationResult,
    PartClassification,
)
from pocket_warehouse.triage.triage import triage


def test_low_confidence(test_config):
    cr = ClassificationResult(
        PartClassification("axle", 0, 0.60, 0, 0.70),
        PartClassification("wheel", 0, 1, 0, 1),
        PartClassification("frame", 0, 1, 0, 1),
        PartClassification("body", 0, 1, 0, 1),
        PartClassification("paint", 0, 1, 0, 1),
        None,
    )
    triage_decision = triage(cr, test_config)
    assert triage_decision.decision == "review"
    assert triage_decision.confidence == 0.60
    assert triage_decision.reason == "Low model confidence"
    assert triage_decision.destination_id == 3


def test_not_profitable(test_config):
    cr = ClassificationResult(
        PartClassification("axle", 4, 1, 2, 1),
        PartClassification("wheel", 4, 1, 2, 1),
        PartClassification("frame", 4, 1, 2, 1),
        PartClassification("body", 4, 1, 2, 1),
        PartClassification("paint", 4, 1, 0, 1),
        None,
    )
    triage_decision = triage(cr, test_config)
    assert triage_decision.decision == "scrap"
    assert triage_decision.confidence == 1.00
    assert triage_decision.reason == "Not profitable"
    assert triage_decision.destination_id == 0


def test_resell_no_damage(test_config):
    cr = ClassificationResult(
        PartClassification("axle", 0, 1, 0, 1),
        PartClassification("wheel", 0, 1, 0, 1),
        PartClassification("frame", 0, 1, 0, 1),
        PartClassification("body", 0, 1, 0, 1),
        PartClassification("paint", 0, 1, 0, 1),
        None,
    )
    triage_decision = triage(cr, test_config)
    assert triage_decision.decision == "resell"
    assert triage_decision.confidence == 1.00
    assert triage_decision.reason == "No repairs needed"
    assert triage_decision.destination_id == 2


def test_profitable_repair(test_config):
    cr = ClassificationResult(
        PartClassification("axle", 2, 1, 1, 1),
        PartClassification("wheel", 0, 1, 0, 1),
        PartClassification("frame", 0, 1, 0, 1),
        PartClassification("body", 0, 1, 0, 1),
        PartClassification("paint", 2, 1, 0, 1),
        None,
    )
    triage_decision = triage(cr, test_config)
    assert triage_decision.decision == "refurbish"
    assert triage_decision.confidence == 1.00
    assert triage_decision.reason == "Repairs needed, profitable to refurbish"
    assert triage_decision.destination_id == 1
