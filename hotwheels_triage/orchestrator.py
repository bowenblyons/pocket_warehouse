from hotwheels_triage.schemas import ClassificationResult
from hotwheels_triage.triage import triage
from vision.capture import SimCamera
from vision.infer import infer

def main():

    cam = SimCamera()
    img = cam.capture()
    result = infer(img_path=img)
    decision = triage(result)
    status = arm_control(decision)
    print(status)
    
if __name__ == "__main__":
    main()
