from pocket_warehouse.hardware.camera import Camera
from pocket_warehouse.hardware.servo import Servo
import time

def capture_image_set(platform_servo_id: int = 5) -> list[str]:
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
