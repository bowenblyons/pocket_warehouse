import time
from pathlib import Path

from pocket_warehouse.hardware.camera import Camera
from pocket_warehouse.hardware.servo import Servo


def modified_capture_image_set(car_id: str, platform_servo_id: int = 3) -> list[str]:
    """Modified version of capture_image_set used for data collection.

    Rotates servo and captures images of cars and puts them in a directory
    named car_id naming the files <car_id>-{0, 75, 150}.jpg
    """
    paths: list[str] = []
    camera: Camera = Camera()
    servo: Servo = Servo()
    directory_path = Path(f"data/captures/car{car_id}")
    try:
        directory_path.mkdir()
        print(f"Created directory '{directory_path}'")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists")
    except Exception as e:
        print(f"ERROR: {e}")

    for angle in [0, 75, 150]:
        print(f"{time.time():.2f} → moving to {angle}")
        servo.set_angle(platform_servo_id, angle)
        time.sleep(7.5)
        print(f"{time.time():.2f} → capturing")
        paths.append(str(camera.capture(f"car{car_id}/{car_id}-{angle}.jpg")))
    # reset servo
    servo.set_angle(platform_servo_id, 90)

    return paths

if __name__ == "__main__":
    i = 0
    image_paths: list[str] = []
    file_path = "dataset_paths.txt"

    try:
        while True:
            input(f"Press enter when car{i} is in position...")
            ids: list[str] = modified_capture_image_set(i)

            for id in ids:
                image_paths.append(id)
    except KeyboardInterrupt:
        print(f"\nEnding data collection. {i+1} image sets collected")
        print(f"Appending files to {file_path}")

    with open(file_path, "w") as file:
        data = '\n'.join(image_paths)

        file.write(data)
    
    print("Great success!")
