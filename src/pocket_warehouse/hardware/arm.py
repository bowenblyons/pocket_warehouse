import time

from pocket_warehouse.hardware.servo import Servo


class Arm:
    """Controls robotic arm for Hot Wheels triage system.

    Manages 3 servos for base rotation, elbow, and claw.
    Provides high level movement operations: pickup_car, scrap, refurbish,
    resell, and review.
    """

    # servo indices
    SERVO_BASE: int = 0
    SERVO_ELBOW: int = 1
    SERVO_CLAW: int = 2

    # array for logging
    SERVO_NAMES: list[str] = ["Base", "Elbow", "Claw"]

    # angles
    BASE_HOME: int = 90
    CLAW_HOME: int = 30
    ELBOW_HOME: int = 10
    POSITION_PICKUP: int = 180
    POSITION_SCRAP: int = 130
    POSITION_REFURBISH: int = 90
    POSITION_RESELL: int = 50
    POSITION_REVIEW: int = 1
    ELBOW_PARTIAL_LOWER: int = 112
    ELBOW_LOWER: int = 140
    CLAW_OPEN: int = CLAW_HOME
    CLAW_CLOSE: int = 130

    def __init__(self):
        self.servo_id: list[int] = [0, 1, 2]  # 3 is platform swivel
        self.servo: Servo = Servo()
        self.__home()

    def __set_angle(self, servo: int, angle: int, delay: float = 7.5) -> None:
        """Set servo angle and wait for movement to complete.

        It's a cheap arm so patience is required...
        """
        self.servo.set_angle(servo, angle)
        print(f"{self.SERVO_NAMES[servo]}: set to {angle}, delay = {delay} ")
        time.sleep(delay)

    def __home(self) -> None:
        """Return all servos to home position"""
        self.__set_angle(self.SERVO_BASE, self.BASE_HOME)
        self.__set_angle(self.SERVO_CLAW, self.CLAW_HOME)
        self.__set_angle(self.SERVO_ELBOW, self.ELBOW_HOME)

    def __close_claw(self) -> None:
        self.__set_angle(self.SERVO_CLAW, self.CLAW_CLOSE)

    def __open_claw(self) -> None:
        self.__set_angle(self.SERVO_CLAW, self.CLAW_OPEN)

    def __position_pickup(self) -> None:
        self.__set_angle(self.SERVO_BASE, self.POSITION_PICKUP)

    def __position_scrap(self) -> None:
        self.__set_angle(self.SERVO_BASE, self.POSITION_SCRAP)

    def __position_refurbish(self) -> None:
        self.__set_angle(self.SERVO_BASE, self.POSITION_REFURBISH)

    def __position_resell(self) -> None:
        self.__set_angle(self.SERVO_BASE, self.POSITION_RESELL)

    def __position_review(self) -> None:
        self.__set_angle(self.SERVO_BASE, self.POSITION_REVIEW)

    def __elbow_partial_lower(self) -> None:
        self.__set_angle(self.SERVO_ELBOW, self.ELBOW_PARTIAL_LOWER)

    def __elbow_lower(self) -> None:
        self.__set_angle(self.SERVO_ELBOW, self.ELBOW_LOWER)

    def pickup_car(self) -> None:
        """Movement sequence for picking up car from platform."""
        self.__position_pickup()
        self.__open_claw()
        self.__elbow_partial_lower()
        self.__elbow_lower()
        self.__close_claw()
        self.__elbow_partial_lower()

    def scrap(self) -> None:
        """Movement sequence for moving car and releasing onto scrap track.

        Must be preceded by pickup_car()
        """
        self.__position_scrap()
        self.__elbow_lower()
        self.__open_claw()
        self.__home()

    def refurbish(self) -> None:
        """Movement sequence for moving car and releasing onto refurbish track.

        Must be preceded by pickup_car()
        """
        self.__position_refurbish()
        self.__elbow_lower()
        self.__open_claw()
        self.__home()

    def resell(self) -> None:
        """Movement sequence for moving car and releasing onto resell track.

        Must be preceded by pickup_car()
        """
        self.__position_resell()
        self.__elbow_lower()
        self.__open_claw()
        self.__home()

    def review(self) -> None:
        """Movement sequence for moving car and releasing onto review track.

        Must be preceded by pickup_car()
        """
        self.__position_review()
        self.__elbow_lower()
        self.__open_claw()
        self.__home()
       
if __name__ == "__main__":
    arm = Arm()
    arm.pickup_car()
