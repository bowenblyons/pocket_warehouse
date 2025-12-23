from dataclasses import dataclass

@dataclass
class ClassificationResult:
    model_guess: str
    damage_score: float
    confidence: float

@dataclass
class TriageDecision:
    route: str
    funnel_id: int
    reason: str

