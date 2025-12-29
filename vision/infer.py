#!/usr/bin/env python3
import numpy as np
import tflite_runtime.interpreter as tf
from PIL import Image

import sys

IMG_SIZE = 192

def load_label_map(path="vision/label_map.txt"):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            i, name = line.strip().split(",", 1)
            m[int(i)] = name
    return m

def preprocess(path, img_size = IMG_SIZE):

    img = Image.open(path).convert('RGB')
    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    img_data = np.array(img, dtype=np.uint8)
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

def main():
    if len(sys.argv) < 3:
        print("Usage: python infer.py model.tflite path/to/image.jpg")
        sys.exit(1)

    model_path = sys.argv[1]
    img_path = sys.argv[2]

    labels = load_label_map()

    interpreter = tf.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    x = preprocess(img_path)

    # handle int8 models
    if in_details["dtype"] == np.int8:
        scale, zero = in_details["quantization"]
        xq = np.round(x / scale + zero).astype(np.int8)
        interpreter.set_tensor(in_details["index"], xq)
    else:
        interpreter.set_tensor(in_details["index"], x.astype(in_details["dtype"]))

    interpreter.invoke()
    y = interpreter.get_tensor(out_details["index"])

    # dequantize output if needed
    if out_details["dtype"] == np.int8:
        scale, zero = out_details["quantization"]
        y = (y.astype(np.float32) - zero) * scale

    probs = y[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    print(f"Prediction: {labels[pred]}  (conf={conf:.3f})")
    print("Probs:", {labels[i]: float(probs[i]) for i in range(len(probs))})

if __name__ == "__main__":
    main()
