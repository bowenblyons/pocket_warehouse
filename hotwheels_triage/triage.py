from .schemas import ClassificationResult, TriageDecision

def triage(result: ClassificationResult) -> TriageDecision:
    if result.confidence < 0.5:
        return TriageDecision(route="MANUAL_REVIEW", funnel_id=0, reason="Low confidence")

    if result.damage_score < 0.3:
        return TriageDecision(route="RESELL", funnel_id=1, reason="Low damage")
    
    if result.damage_score < 0.7:
        return TriageDecision(route="REFURBISH", funnel_id=2, reason="Moderate damage")
    
    return TriageDecision(route="SCRAP", funnel_id=3, reason="Severe damage")