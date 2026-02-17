import time

from pocket_warehouse.hardware.camera import Camera
from pocket_warehouse.hardware.servo import Servo


def capture_image_set(platform_servo_id: int = 5) -> list[str]:
    """Handles capturing three consecutive images of car at different angles.

    Manages a single servo to control platform rotation and the camera module.
    """
    paths: list[str] = []
    camera: Camera = Camera()
    servo: Servo = Servo()

    for angle in [0, 75, 150]:
        print(f"{time.time():.2f} → moving to {angle}")
        servo.set_angle(platform_servo_id, angle)
        time.sleep(1)
        print(f"{time.time():.2f} → capturing")
        paths.append(str(camera.capture(f"image{angle}.jpg")))
    return paths

if __name__ == "__main__":
    print(capture_image_set())
