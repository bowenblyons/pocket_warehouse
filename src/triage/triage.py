from schemas import ClassificationResult, TriageDecision

def Triage(result: ClassificationResult) -> TriageDecision:

    return TriageDecision(result="review", destination_id=0, reason="Low confidence score.")
