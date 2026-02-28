from PIL import Image
from collections import defaultdict

from pocket_warehouse.hardware.capture_images import capture_image_set
from pocket_warehouse.inference.model import PocketWarehouseInference
from pocket_warehouse.triage.triage import triage
from pocket_warehouse.hardware.arm import Arm
from pocket_warehouse.schemas.schemas import ClassificationResult, PartClassification
from pocket_warehouse.utils.config import load_config

def main() -> None:
    # order of operations when run:
    # 1. take pics with capture_images
    #     - will take 3 pics
    # 2. run inference on images
    #     - will need to run 3 times
    #     - will need to combine results
    # 3. run inference through triage system
    # 4. give appropriate commands to arm to move the car

    # get set of three images of car
    image_paths: list[str] = capture_image_set()

    model = PocketWarehouseInference("models/pocket_warehouse.tflite")
    predictions = []
    
    for path in image_paths:
        image = Image.open(path)
        predictions.append(model.predict_with_confidence(image))

    combined_pred = combine_predictions(predictions)
    print("Combined_prediction: ", combined_pred)
    classification_result = get_classification_result(combined_pred)

    cfg = load_config()
    
    triage_result = triage(classification_result, cfg)

    print(triage_result.decision)

    arm = Arm()
    arm.pickup_car()

    if (triage_result.decision == "scrap"):
        arm.scrap()
    elif (triage_result.decision == "refurbish"):
        arm.refurbish()
    elif (triage_result.decision == "resell"):
        arm.resell()
    else: # triage_result.decision == "review"
        # will also trigger review for unforseen exceptions
        arm.review()

def get_classification_result(pred: dict) -> ClassificationResult:
    axle = PartClassification("axle", pred["axle"]["severity"]["class"], pred["axle"]["severity"]["confidence"], pred["axle"]["functional"]["class"], pred["axle"]["functional"]["confidence"])
    wheel = PartClassification("wheel", pred["wheel"]["severity"]["class"], pred["wheel"]["severity"]["confidence"], pred["wheel"]["functional"]["class"], pred["wheel"]["functional"]["confidence"])
    frame = PartClassification("frame", pred["frame"]["severity"]["class"], pred["frame"]["severity"]["confidence"], pred["frame"]["functional"]["class"], pred["frame"]["functional"]["confidence"])
    body = PartClassification("body", pred["body"]["severity"]["class"], pred["body"]["severity"]["confidence"], pred["body"]["functional"]["class"], pred["body"]["functional"]["confidence"])
    paint = PartClassification("paint", pred["paint"]["severity"]["class"], pred["paint"]["severity"]["confidence"], pred["paint"]["functional"]["class"], pred["paint"]["functional"]["confidence"])
    return ClassificationResult(axle, wheel, frame, body, paint, None)

def combine_predictions(predictions: list[dict]) -> dict:
    combined = {}

    parts = predictions[0].keys()

    for part in parts:
        combined[part] = {}

        for label_type in ["severity", "functional"]:
            
            max_conf = 0.0
            max_conf_class = -1
            for p in predictions:
                pred = p[part][label_type]
                if pred["confidence"] > max_conf:
                    max_conf = pred["confidence"]
                    max_conf_class = pred["class"]

            combined[part][label_type] = {
                "class": max_conf_class,
                "confidence": max_conf
            }

    return combined

if __name__ == "__main__":
    main()
