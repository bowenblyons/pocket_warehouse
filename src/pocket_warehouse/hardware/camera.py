from datetime import datetime
from pathlib import Path


class Camera:
    """Controls camera for hot wheels triage system.

    Manages a single camera module on the raspberry pi,
    captures an image and puts it in the data/captures directory.
    """

    def __init__(self, data_dir: Path = Path("data/captures")) -> None:
        from picamera2 import Picamera2

        self.data_directory: Path = data_dir
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.camera: Picamera2 = Picamera2()

        config = self.camera.create_still_configuration()
        self.camera.configure(config)
        self.camera.start()

    def capture(self, filename: str | None) -> Path:
        """Captures an image and returns its path."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"image{timestamp}.jpg"

        filepath = self.data_directory / filename
        self.camera.capture_file(str(filepath))
        return filepath

    def close(self) -> None:
        self.camera.close()


def main() -> None:
    cam = Camera()
    cam.capture("test.jpg")
    cam.close()


if __name__ == "__main__":
    main()
