import onnxruntime as ort # type: ignore
from torchvision import transforms # type: ignore

class PocketWarehouseModel():
    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name 
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def infer(self, image):
        x = self.transform(image).unsqueeze(0).numpy()

        sev_flat, fun_flat = self.session.run(
            self.output_names,
            {self.input_name: x}
        )

        sev_logits = sev_flat.reshape(5, 5)
        fun_logits = fun_flat.reshape(5, 3)

        return {
            "severity_logits": sev_logits,
            "functional_logits": fun_logits,
            "severity": sev_logits.argmax(axis=1),
            "functional": fun_logits.argmax(axis=1),
        }
