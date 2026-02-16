from dataclasses import dataclass
from datetime import datetime, timezone

################################################
########## CLASSIFICATION CLASSES ##############
################################################

@dataclass
class PartClassification:
    name: str
    severity: int
    severity_confidence: float
    functional: int
    functional_confidence: float

    def __post_init__(self):
        """Validate confidence scores"""
        if not 0.0 <= self.severity_confidence <= 1.0:
            raise ValueError(f"severity_confidence must be 0-1, got {self.severity_confidence}")
        if not 0.0 <= self.functional_confidence <= 1.0:
            raise ValueError(f"functional_confidence must be 0-1, got {self.functional_confidence}")

    @property
    def min_confidence(self) -> float:
        """Get the lower of functional and severity confidence score"""
        return min(self.severity_confidence, self.functional_confidence)

@dataclass
class ClassificationResult:
    axles: PartClassification
    wheels: PartClassification
    frame: PartClassification
    body: PartClassification
    paint: PartClassification
    timestamp: datetime | None

    def __post_init__(self):
        if self.timestamp == None:
            self.timestamp = datetime.now(tz=timezone.utc)

    def get_parts(self) -> list[PartClassification]:
        """Get dictionary of parts"""
        return [self.axles, self.wheels, self.frame, self.body, self.paint]

    @property
    def min_confidence(self) -> float:
        """Get the lowest confidence score from the parts"""
        return min(part.min_confidence for part in self.get_parts())
    
#############################################
######## FIN AND DATABASE ###################
#############################################

@dataclass
class PartRequirement:
    name: str
    action: str
    cost: float
    is_stocked: bool

@dataclass
class FinancialAnalysis:
    total_cost: float
    expected_resale: float

    @property
    def margin(self) -> float:
        """Total expected profit margin"""
        return self.expected_resale - self.total_cost

    @property
    def roi_percent(self) -> float:
        """Total expected return on investment (ROI)"""
        if self.total_cost > 0:
            return (self.margin / self.total_cost ) * 100
        return 0.0

    @property
    def is_profitable(self) -> bool:
        """Return true if there is a profit margin"""
        return self.margin > 0

@dataclass
class InventoryImpact:
    parts_needed: dict[str, int]
    remaining: dict[str, int]
    reorder_triggered: list[str]

@dataclass
class TriageDecision:
    decision: str
    reason: str
    confidence: float
    financial: FinancialAnalysis | None
    work_order: list[PartRequirement] | None
    inventory_impact: InventoryImpact | None
    destination_id: int | None # 0 - scrap, 1 - refurb, 2 - resell, 3 - review
    timestamp: datetime | None
    classification_input: ClassificationResult | None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(tz=timezone.utc)

    # method to export it to json/dictionary
