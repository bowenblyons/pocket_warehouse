from pocket_warehouse.hardware.servo import Servo
import time

class Arm():
    """Controls robotic arm for Hot Wheels triage system.

    Manages 5 servos for base rotation, shoulder, elbow, wrist, and claw.
    Provides high level movement operations: pickup_car, scrap, refurbish,
    resell, and review.
    """

    # servo indices
    SERVO_BASE: int = 0
    SERVO_SHOULDER: int = 1
    SERVO_ELBOW: int = 2
    # wrist servo is bad: SERVO_WRIST: int = 3     
    SERVO_CLAW: int = 4

    # angles
    HOME: int = 90
    POSITION_PICKUP: int = 0
    POSITION_SCRAP: int = 45
    POSITION_REFURBISH: int = 90
    POSITION_RESELL: int = 135
    POSITION_REVIEW: int = 180
    ELBOW_PRELOWER: int = 40
    SHOULDER_LOWER: int = 20
    CLAW_OPEN: int = 140
    CLAW_CLOSE: int = 30
    servo_id: list[int]
    servo: Servo
    
    def __init__(self):
        servo_id: list[int] = [0, 1, 2, 3, 4] # 5 is platform swivel
        servo: Servo = Servo()
        self.__home()

    def __set_angle(self, servo: int, angle: int, delay: float = 2.0) -> None:
        """Set servo angle and wait for movement to complete."""
        self.servo.set_angle(servo, angle)
        time.sleep(delay)

    def __home(self) -> None:
        """Return all servos to home position"""
        for i in self.servo_id:
            self.__set_angle(i, self.HOME)
            
    def __close_claw(self) -> None:
        self.__set_angle(self.SERVO_CLAW, self.CLAW_CLOSE)

    def __open_claw(self) -> None:
        self.__set_angle(self.SERVO_CLAW, self.CLAW_OPEN)

    def __arm_prelower(self) -> None:
        self.__set_angle(self.SERVO_ELBOW, self.ELBOW_PRELOWER)

    def __arm_lower(self) -> None:
        self.__set_angle(self.SERVO_SHOULDER, self.SHOULDER_LOWER)

    def __arm_raise(self) -> None:
        self.__set_angle(self.SERVO_SHOULDER, self.HOME)

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

    def pickup_car(self) -> None:
        """Movement sequence for picking up car from platform."""
        self.__position_pickup()
        self.__arm_prelower()
        self.__open_claw()
        self.__arm_lower()
        self.__close_claw()
        self.__arm_raise()

    def scrap(self) -> None:
        """Movement sequence for moving car and releasing onto scrap track.

        Must be preceded by pickup_car()
        """
        self.__position_scrap()
        self.__arm_lower()
        self.__open_claw()
        self.__home()

    def refurbish(self) -> None:
        """Movement sequence for moving car and releasing onto refurbish track.

        Must be preceded by pickup_car()
        """
        self.__position_refurbish()
        self.__arm_lower()
        self.__open_claw()
        self.__home()

    def resell(self) -> None:
        """Movement sequence for moving car and releasing onto resell track.

        Must be preceded by pickup_car()
        """
        self.__position_resell()
        self.__arm_lower()
        self.__open_claw()
        self.__home()

    def review(self) -> None:
        """Movement sequence for moving car and releasing onto review track.

        Must be preceded by pickup_car()
        """
        self.__position_review()
        self.__arm_lower()
        self.__open_claw()
        self.__home()
