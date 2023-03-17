import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
armnn_delegate = tf.lite.experimental.load_delegate(
    library="<path-to-armnn-binaries>/libarmnnDelegate.so",
    options={
        "backends": "CpuAcc,GpuAcc,CpuRef",
        "logging-severity": "info"
    }
)

# Delegates/Executes all operations supported by Arm NN to/with Arm NN
interpreter = tf.lite.Interpreter(
    model_path="./models/ocr_model.tflite",
    experimental_delegates=[armnn_delegate]
)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run model
interpreter.invoke()

# Print out result
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
