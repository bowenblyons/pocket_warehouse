from pathlib import Path
from datetime import datetime

class Camera():

	def __init__(self, data_dir: Path = Path("data/captures")) -> None:
		from picamera2 import Picamera2 # type: ignore

		self.data_directory: Path = data_dir
		self.data_directory.mkdir(parents=True, exist_ok=True)
		self.camera: Picamera2 = Picamera2()

		config = self.camera.create_still_configuration() # type: ignore
		self.camera.configure(config) # type: ignore
		self.camera.start() # type: ignore

	def capture(self, filename: str | None) -> Path:
		if filename is None:
			timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
			filename = f"image{timestamp}.jpg"

		filepath = self.data_directory / filename
		self.camera.capture_file(str(filepath)) # type: ignore
		return filepath

	def close(self) -> None:
		self.camera.close() # type: ignore

def main() -> None:
	cam = Camera()
	cam.capture("test.jpg")
	cam.close()

if __name__ == "__main__":
	main()
