import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from board import SCL, SDA


class Servo:
    """Interface for controlling servos via PCA9685 I2C PWM controller."""

    def __init__(self) -> None:
        self.i2c: busio.I2C = busio.I2C(SCL, SDA)
        self.pca: PCA9685 = PCA9685(self.i2c, address=0x40)
        self.pca.frequency = 50

    def set_angle(self, id: int, angle: int) -> None:
        """Set servo angle.

        Args:
            id: Servo channel (0-15 on PCA9685)
            angle: Target angle in degrees (0-180)
        """
        servo_angle = servo.Servo(
            self.pca.channels[id],
            min_pulse=500,
            max_pulse=2400,
            actuation_range=180,
        )  # type: ignore
        servo_angle.angle = angle
