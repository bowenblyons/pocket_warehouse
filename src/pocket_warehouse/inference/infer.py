from pocket_warehouse.schemas.schemas import (
    ClassificationResult,
    Part,
    SeverityLevel,
    FunctionalLevel,
)
from datetime import datetime


def sample_infer() -> ClassificationResult:
    """Example inference output for testing, returns hardcoded result."""
    axle = Part(
        SeverityLevel.NO_DAMAGE, 0.49, FunctionalLevel.NO_IMAIRMENT, 0.72
    )
    wheels = Part(
        SeverityLevel.NO_DAMAGE, 0.71, FunctionalLevel.NO_IMAIRMENT, 0.94
    )
    frame = Part(
        SeverityLevel.NO_DAMAGE, 0.95, FunctionalLevel.NO_IMAIRMENT, 0.95
    )
    body = Part(
        SeverityLevel.MISSING, 0.68, FunctionalLevel.NON_FUNCTIONAL, 0.99
    )
    paint = Part(SeverityLevel.MINOR, 0.88, FunctionalLevel.NO_IMAIRMENT, 0.99)

    return ClassificationResult(
        axle, wheels, frame, body, paint, datetime.now()
    )
