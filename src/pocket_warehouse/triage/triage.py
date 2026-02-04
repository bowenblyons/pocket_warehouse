from pocket_warehouse.schemas.schemas import ClassificationResult, PartClassification, TriageDecision
from pocket_warehouse.utils.config import get_config

def Triage(result: ClassificationResult) -> TriageDecision:

    cfg = get_config()
    triage_result: str
    
    # check confidence
    if result.min_confidence < 0.60:
        triage_result = "review"    

    
    # for each part calculate repair requirements
    parts = result.get_parts()
    need_part: list[bool]

    for part in parts:
        if part.severity == 4 or part.functional == 2:
            
        elif part.severity >= 2 or part.functional >= 1:
            
        
    # check inventory available
    # scrap rules
    # calculate repair financials
    # decision
    # reserve inventory if needed
    # create work order
