from dataclasses import dataclass

@dataclass
class ClassificationResult:
    model_guess: str
    confidence: float
    low: float
    moderate: float
    severe: float

@dataclass
class TriageDecision:
    route: str
    funnel_id: int
    reason: str

