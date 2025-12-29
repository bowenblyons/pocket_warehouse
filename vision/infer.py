import os

import numpy as np
import tensorflow as tf
from PIL import Image

TFLITE_PATH = os.environ.get("TFLITE_PATH", "model_int8.tflite")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
# Must match folder names order used during training (alphabetical by default)
CLASS_NAMES = ["BROKEN", "OK", "WORN"]  # adjust if your directory names differ


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.uint8)  # uint8 because we exported uint8 input
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python infer_tflite.py path/to/image.jpg")
        raise SystemExit(2)

    image_path = sys.argv[1]
    x = load_image(image_path)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details["index"])[0]  # uint8 logits/probs-ish

    # Convert uint8 output to float-ish probabilities
    # If you want exact calibration, inspect output_details['quantization']
    scale, zero = output_details["quantization"]
    y_f = (y.astype(np.float32) - zero) * scale

    pred = int(np.argmax(y_f))
    conf = float(np.max(tf.nn.softmax(y_f).numpy()))

    print({"label": CLASS_NAMES[pred], "confidence": round(conf, 4)})


if __name__ == "__main__":
    main()
