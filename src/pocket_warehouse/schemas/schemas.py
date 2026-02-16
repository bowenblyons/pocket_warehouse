from dataclasses import dataclass
from datetime import UTC, datetime

################################################
########## CLASSIFICATION CLASSES ##############
################################################


@dataclass
class PartClassification:
    """Represents a the models classification for a single part."""

    name: str
    severity: int
    severity_confidence: float
    functional: int
    functional_confidence: float

    def __post_init__(self):
        """Validate confidence scores"""
        if not 0.0 <= self.severity_confidence <= 1.0:
            sev_con = self.severity_confidence
            raise ValueError(f"severity_confidence must be 0-1, got {sev_con}")
        if not 0.0 <= self.functional_confidence <= 1.0:
            fun_con = self.functional_confidence
            raise ValueError(
                f"functional_confidence must be 0-1, got {fun_con}"
            )

    @property
    def min_confidence(self) -> float:
        """Get the lower of functional and severity confidence score"""
        return min(self.severity_confidence, self.functional_confidence)


@dataclass
class ClassificationResult:
    """ "Represents the classification of all parts on the vehicle."""

    axle: PartClassification
    wheel: PartClassification
    frame: PartClassification
    body: PartClassification
    paint: PartClassification
    timestamp: datetime | None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(tz=UTC)

    def get_parts(self) -> list[PartClassification]:
        """Get dictionary of parts"""
        return [self.axle, self.wheel, self.frame, self.body, self.paint]

    @property
    def min_confidence(self) -> float:
        """Get the lowest confidence score from the parts"""
        return min(part.min_confidence for part in self.get_parts())


#############################################
######## FIN AND DATABASE ###################
#############################################


@dataclass
class PartRequirement:
    """Info about a single part relevant for triage system.

    name - part name,
    action - resell|refurbish|scrap|review,
    cost - cost to repair,
    is_stocked - true if the part is in stock
    """

    name: str
    action: str
    cost: float
    is_stocked: bool


@dataclass
class FinancialAnalysis:
    """Info about the financial viability of the repair
    based on the expected resale value."""

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
            return (self.margin / self.total_cost) * 100
        return 0.0

    @property
    def is_profitable(self) -> bool:
        """Return true if there is a profit margin"""
        return self.margin > 0


@dataclass
class InventoryImpact:
    """Calculated impact to current inventory
    used to update database in future iteration."""

    parts_needed: dict[str, int]
    remaining: dict[str, int]
    reorder_triggered: list[str]


@dataclass
class TriageDecision:
    """The decision made by the triage system
    along with information for json log output."""

    decision: str
    reason: str
    confidence: float
    financial: FinancialAnalysis | None
    work_order: list[PartRequirement] | None
    inventory_impact: InventoryImpact | None
    destination_id: int | None  # 0 - scrap, 1 - refurb, 2 - resell, 3 - review
    timestamp: datetime | None
    classification_input: ClassificationResult | None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(tz=UTC)

    # method to export it to json/dictionary
