import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

CAR_PARTS = ["axle", "wheel", "frame", "body", "paint"]


class PocketWarehouseInference:
    def __init__(self, model_path: str) -> None:
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_dtype = self.input_details[0]["dtype"]
        self.input_scale, self.input_zero_point = self.input_details[0][
            "quantization"
        ]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for inference"""
        img = image.resize((224, 224))
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.299, 0.244, 0.255]

        img = img / self.input_scale + self.input_zero_point
        img = np.clip(img, 0, 255).astype(np.uint8)

        img = np.expand_dims(img, axis=0)
        return img

    def dequantize_output(
        self, quantized_output: np.ndarray, output_idx: int
    ) -> np.ndarray:
        scale, zero_point = self.output_details[output_idx]["quantization"]
        return (quantized_output.astype(np.float32) - zero_point) * scale

    def predict(self, image: Image.Image) -> dict[str, np.ndarray]:
        """Run inference on a single image."""
        img = self.preprocess(image)

        self.interpreter.set_tensor(self.input_details[0]["index"], img)
        self.interpreter.invoke()

        # Get both outputs
        severity_raw = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )
        functional_raw = self.interpreter.get_tensor(
            self.output_details[1]["index"]
        )

        # Dequantize if needed
        severity = self.dequantize_output(severity_raw, 0)
        functional = self.dequantize_output(functional_raw, 1)

        # Shape: (1, 5, num_classes) -> (5, num_classes)
        severity = severity[0]
        functional = functional[0]

        return {
            "severity": np.argmax(severity, axis=-1),
            "functional": np.argmax(functional, axis=-1),
            "severity_logits": severity,
            "functional_logits": functional,
        }

    def predict_with_confidence(self, image: Image.Image) -> dict:
        """Run inference and return predictions with confidence scores."""
        predictions = self.predict(image)

        # Apply softmax to get probabilities
        sev_probs = self._softmax(predictions["severity_logits"])
        fun_probs = self._softmax(predictions["functional_logits"])

        result = {}
        for i, part in enumerate(CAR_PARTS):
            result[part] = {
                "severity": {
                    "class": int(predictions["severity"][i]),
                    "confidence": float(
                        sev_probs[i, predictions["severity"][i]]
                    ),
                },
                "functional": {
                    "class": int(predictions["functional"][i]),
                    "confidence": float(
                        fun_probs[i, predictions["functional"][i]]
                    ),
                },
            }

        return result

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each set of scores."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    model = PocketWarehouseInference("models/pocket_warehouse.tflite")

    # Simple prediction
    image = Image.open("data/sample_images/severe/car03_0011.jpg")
    predictions = model.predict(image)

    print("\nPredictions (class indices):")
    print("Severity:", predictions["severity"])
    print("Functional:", predictions["functional"])

    # Detailed prediction with confidence
    detailed = model.predict_with_confidence(image)

    print("\nDetailed predictions:")
    for part, preds in detailed.items():
        print(f"\n{part.upper()}:")
        print(
            f"  Severity: class {preds['severity']['class']} "
            f"(confidence: {preds['severity']['confidence']:.2%})"
        )
        print(
            f"  Functional: class {preds['functional']['class']} "
            f"(confidence: {preds['functional']['confidence']:.2%})"
        )
