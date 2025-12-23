from hotwheels_triage.schemas import ClassificationResult
from hotwheels_triage.triage import triage

def main():
    mock_result = ClassificationResult( model_guess="F-150", damage_score=0.45, confidence=0.85 )
    decision = triage(mock_result)
    print(decision)

if __name__ == "__main__":
    main()
