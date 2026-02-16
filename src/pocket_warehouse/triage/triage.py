from pocket_warehouse.schemas.schemas import (
    ClassificationResult,
    TriageDecision,
    PartRequirement,
    FinancialAnalysis,
)
from pocket_warehouse.utils.config import get_config
from pocket_warehouse.triage.inventory import check_availability, get_impact
from pocket_warehouse.triage.margin import calculate_cost


def Triage(result: ClassificationResult) -> TriageDecision:

    cfg = get_config()

    # for each part calculate repair requirements
    parts = result.get_parts()
    parts_list_needs: list[PartRequirement] = []
    # assign actions to parts
    for part in parts:
        action = "none"  # professor always told me to hope for the best
        name = part.name
        availability = False  # assume unavailable
        cost = -1.00  # cost calculated later

        if part.severity == 4 or part.functional == 2:
            action = "replace"
        elif part.severity >= 2 or part.functional >= 1:
            action = "repair"

        parts_list_needs.append(
            PartRequirement(name, action, cost, availability)
        )

    # check availability
    for part_req in parts_list_needs:
        part_req.is_stocked = check_availability(part_req.name, 1)

    # compute financials
    fin = FinancialAnalysis(0.0, cfg.resell_market_value)

    for part_req in parts_list_needs:
        part_req.cost = calculate_cost(
            part_req.name, part_req.action, part_req.is_stocked
        )
        fin.total_cost += part_req.cost

    # decision: scrap if its not profitable, resell if no repairs, refurb if repairs, review if low confidence from model
    if result.min_confidence < cfg.confidence_threshold:
        triage_result = "review"
        confidence = result.min_confidence
        reason = "Low model confidence"
        destination = 3
    elif not fin.is_profitable:
        triage_result = "scrap"
        confidence = 1.00
        reason = "Not profitable"
        destination = 0
    elif fin.total_cost == 0:
        triage_result = "resell"
        confidence = 1.00
        reason = "No repairs needed"
        destination = 2
    else:
        triage_result = "refurbish"
        confidence = 1.00
        reason = "Repairs needed, profitable to refurbish"
        destination = 1

    return TriageDecision(
        triage_result,
        reason,
        confidence,
        fin,
        parts_list_needs,
        get_impact(parts_list_needs),
        destination,
        None,
        result,
    )
