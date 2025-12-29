from hotwheels_triage.schemas import ClassificationResult
from hotwheels_triage.triage import triage

def test_low_damage_goes_to_resell():
    result = ClassificationResult(model_guess="Mustang", confidence=0.9, damage_score=0.2)
    decision = triage(result)
    assert decision.route == "RESELL"
    assert decision.funnel_id == 1
    assert decision.reason == "Low damage"