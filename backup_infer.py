import numpy as np
import cv2
import time
import tflite_runtime.interpreter as tflite

IMG_SIZE = 192

def main():
    model_path = "vision/model_int8.tflite"
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']

    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]

    input_dtype = input_details[0]['dtype']

    image = cv2.imread("dataset/car15_0001.jpg")
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_data_float = np.expand_dims(image, axis=0)  # Add batch dimension
    input_data_float = input_data_float.astype(np.float32)

    if input_dtype == np.int8 or input_dtype == np.uint8:
        # Quantize the input: q = (f / scale) + zero_point
        input_data = (input_data_float / input_scale) + input_zero_point
        input_data = np.round(input_data).astype(input_dtype)
    else:
        input_data = input_data_float

    # --- 5. Run Inference ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    # --- 6. Get and Dequantize Output ---
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # If output is quantized, convert back to float for human-readable results
    output_scale, output_zero_point = output_details[0]['quantization']
    
    if output_scale > 0: # Ensure quantization parameters exist
        # Dequantize: f = (q - zero_point) * scale
        output_data_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
        print("Dequantized Output:", output_data_float)
    else:
        print("Raw Output:", output_data)

if __name__ == "__main__":
    main()
