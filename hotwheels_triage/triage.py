from hotwheels_triage.schemas import ClassificationResult, TriageDecision

def triage(result: ClassificationResult) -> TriageDecision:
    if result.confidence < 0.5:
        return TriageDecision(route="review", funnel_id=0, reason="Low confidence")

    if result.model_guess == "low":
        return TriageDecision(route="resell", funnel_id=1, reason="Low damage")

    if result.model_guess == "moderate":
        return TriageDecision(route="refurbish", funnel_id=2, reason="Moderate damage")
    
    return TriageDecision(route="scrap", funnel_id=3, reason="Severe damage")
