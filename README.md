# Pocket Warehouse
Pocket Warehouse is a Raspberry Pi powered robotic triage system that classifies Hot Wheels cars by damage severity and functional impairment using computer vision and automatically sorts them using a servo driven robotic arm.

## Demo
Partial Success! Need to make some adjustments to the arm to better distribute the weight.
[![Partial Success]](https://raw.githubusercontent.com/bowenblyons/pocket_warehouse/main/demo_video/partial_success.mp4)

The system captures a set of three images of the car at different angles, performs inference on each image, merges the predictions by selecting for max confidence, calculates the potential profit, and instructs a robotic arm to move the car to the appropriate location.

## Motivation
Automated triage systems are commonly used in refurbishment and resale operations to process incoming products like phones, laptops, and cameras. These systems evaluate the item, estimate the repair cost, and determine how it should be processed.

I built Pocket Warehouse to explore implementing a triage workflow on a small embedded system. The project focuses on designing a full pipeline, from image capture and machine learning inference to decision logic and robotic sorting.

## System Architecture
1. Camera and Intake Platform
    - Gets three images of the car at three different angles.
2. TFLite Inference
    - Runs inference on the three images independently.
3. Inference Post-processing
    - Predications are merged by selecting classifications with the highest confidence.
4. Business Logic
    - Calculates potential profit.
    - Applies configurable conditions for the classification.
    - Sets the cars destination as one of scrap, refurbish, resell, or human review.
5. Robot Arm
    - Programmed servos to pick up car from intake platform and move it to one of the four locations.

## Features

- Computer vision component level damage and functionality classification.
- Multi-view inference with max confidence fusion.
- Configurable market value, labor cost, per part cost and inventory, and confidence threshold via YAML.
- Embedded ML run locally on a Raspberry Pi using a ResNet18 backbone fine tuned with a dataset created from my sons Hot Wheels collection.
- Cost based repair decision logic.
- Servo controlled robotic arm sorting.

## Directory Structure
```
pocket_warehouse/
├── config/
├── data/
│   ├── captures/
│   └── sample_images/
├── models/
├── src/
│   └── pocket_warehouse/
│       ├── hardware/
│       ├── inference/
│       ├── schemas/
│       ├── triage/
│       └── utils/
└── tests/
```
## Lessons Learned
- ML conversion pipelines are messy unless you find the right tools (pip install ai-edge-litert).
- Dependencies can be simplified on an embedded device with a single purpose, no virtual environments needed if system Python only has one job.
- KISS and get it done.
- How to fine tune a model to run a component level multi-task classification.

Training script lives at:
https://github.com/bowenblyons/pocket_warehouse_model.git
