import tensorflow as tf
import numpy as np
import cv2

from generate_ocr import result_arr
from segment import pad_resize


def main(lite_model_path="./models/tf_lite_ocr/ocr_model.tflite"):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=lite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    test_img = pad_resize(cv2.imread("./test_images/letter_c.png", cv2.IMREAD_GRAYSCALE))
    cv2.imshow("test", test_img)
    cv2.waitKey(0)
    interpreter.set_tensor(input_details[0]['index'], [test_img])

    # Run the model
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Prediction: {result_arr[np.argmax(output_data)]}")


if __name__ == "__main__":
    tf2tflite()
