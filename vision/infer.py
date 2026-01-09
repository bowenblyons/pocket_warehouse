import numpy as np
import tflite_runtime.interpreter as tf
from PIL import Image

import sys

from hotwheels_triage.schemas import ClassificationResult

IMG_SIZE = 192
MODEL_PATH = "model/model_int8.tflite"

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

def infer(img_path, model=MODEL_PATH):

    labels = load_label_map()

    interpreter = tf.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    x = preprocess(img_path)
    
    scale, zero = in_details["quantization"]
    xq = np.round(x / scale + zero).astype(np.int8)
    interpreter.set_tensor(in_details["index"], xq)

    interpreter.invoke()
    y = interpreter.get_tensor(out_details["index"])

    scale, zero = out_details["quantization"]
    y = (y.astype(np.float32) - zero) * scale

    probs = y[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    result = ClassificationResult(
        model_guess = labels[pred],
        confidence = conf,
        low = probs[0],
        moderate = probs[1],
        severe = probs[2]
    )
    
    return result
