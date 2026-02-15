from pocket_warehouse.schemas.schemas import ClassificationResult, TriageDecision, PartRequirement, PARTS_LIST
from pocket_warehouse.utils.config import get_config
from pocket_warehouse.triage.inventory import check_availability

def Triage(result: ClassificationResult) -> TriageDecision:

    cfg = get_config()
    triage_result: str
    
    # check confidence
    if result.min_confidence < 0.60:
        triage_result = "review"    

    
    # for each part calculate repair requirements
    parts = result.get_parts()
    parts_list_needs: list[PartRequirement] = []

    for i in range(len(parts) - 1):
        
        if parts[i].severity == 4 or parts[i].functional == 2:
            action = "replace"

        elif parts[i].severity >= 2 or parts[i].functional >= 1:
            action = "repair"

        else:
            continue

        name = parts[i].name
        availablility = check_availability(name, 1)
        cost = calculate_cost(name, action)
        
        parts_list_needs.append(PartRequirement(name, action, cost, availablility))

    # scrap rules
    # calculate repair financials
    # decision
    # reserve inventory if needed
    # create work order
