from hotwheels_triage.schemas import TriageDecision

class Arm():
    def arm_control(self):
        raise NotImplementedError

class SimArm(Arm):
    def arm_control(self, decision):
        if (decision.)
