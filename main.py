import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import generate_ocr
import segment
from generate_ocr import result_arr, input_size


def tf2tflite(load_filepath="./models/ocr", save_filepath="./models/tf_lite_ocr/"):
    model = tf.keras.models.load_model(load_filepath)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    lite_model = converter.convert()

    with open(save_filepath + "ocr_model.tflite", 'wb') as f:
        f.write(lite_model)


def is_raspberrypi():  # Random function from stack overflow that checks if the running device is a Raspberry Pi
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower():
                return True
    except (Exception,):
        pass
    return False


def camera_input():
    # Define camera input
    vid = cv2.VideoCapture(0)

    # Display the camera output at the specified framerate
    while True:

        # Capture a video frame
        ret, frame = vid.read()

        # Check to make sure something was captured
        if frame is None:
            print("Could not capture from video source!")
            exit(1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)

        # Trigger display
        u_input = cv2.waitKey(1000)

        # Check for user input
        if u_input == ord('c') or u_input == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    if u_input == ord('c'):
        return gray
    else:
        return np.zeros((1, 1, 1), dtype="uint8")


def display_results(input_data, prediction_data):
    # Remove any data with a prediction of less than 50%
    filtered_list = [x for x in zip(input_data, prediction_data) if np.max(x[1]) > 0.5]

    # Display each character and its predicted value
    for idx, data_point in enumerate(filtered_list):

        x_pos = [20, 320, 620, 920, 1220]

        pred = data_point[1]
        window_name = f"{result_arr[np.argmax(pred)]} - {np.max(pred) * 100:0.2f}% - {idx}, "

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, x_pos[idx % 5], 280)
        resize_img = cv2.resize(data_point[0], (280, 280))
        cv2.imshow(window_name, resize_img)
        cv2.waitKey(1)

        if idx % 5 == 4:
            if cv2.waitKey(0) == ord('q'):
                exit(0)
            cv2.destroyAllWindows()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(new_model, epochs, camera, img_path, tflite_model_location):
    if new_model:
        model_path = "./models/ocr"
        generate_ocr.generate_ocr_model(filepath=model_path, epochs=epochs)
        tf2tflite(load_filepath=model_path, save_filepath=tflite_model_location)

    if camera:
        # Get image from camera
        test_image = camera_input()
        if test_image.any():
            print("Capture received!")
        else:
            print("Quit input detected, Goodbye!")
            exit(0)
    else:
        # Load image from path
        if not Path(img_path).is_file():
            print("File not found! Make sure the file name and extension are spelt correctly.")
            exit(1)

        # Load an image in grayscale format
        test_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image into a list of characters in image format
    characters = segment.letters_extract(test_image)

    # Optional use of ArmNN Library
    arm_nn_delegate = None
    if is_raspberrypi():
        arm_nn_delegate = tf.lite.experimental.load_delegate(
            library="",                                                         # TODO: Install library on raspberry pi
            options={
                "backends": "CpuAcc,GpuAcc,CpuRef",
                "logging-severity": "info"
            }
        )

    # Load TFLite model and set the input size to the number of characters
    interpreter = tf.lite.Interpreter(
        model_path=tflite_model_location,
        experimental_delegates=arm_nn_delegate
    )
    interpreter.resize_tensor_input(0, [len(characters), input_size, input_size, 1])
    interpreter.allocate_tensors()

    # Get input and output info
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Load the model with characters for prediction
    interpreter.set_tensor(input_details['index'], characters)

    # Run the model
    interpreter.invoke()

    # Get prediction output data
    prediction = interpreter.get_tensor(output_details['index'])

    # Display the data
    display_results(characters, prediction)


def user_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-image-path", default=None, help="A path to an image to process")
    parser.add_argument("-m", "--model-path", default=None, help="A path to the model.tflite file")
    parser.add_argument("-c", "--camera", default=False, help="Enable webcam input (False by default)")
    parser.add_argument("-n", "--new-model", default=False, help="Generate a new TF model (False by default)")
    parser.add_argument("-e", "--epochs", default=2,
                        help="Specified number of epochs to run when generating a new model")

    args = parser.parse_args()

    if args.model_path is None:
        print("Please provide a model.tflite file!")
        exit(1)

    if args.input_image_path is None and args.camera is False:
        print("Please provide an image file or enable the webcam input!")
        exit(1)

    lite_model_path = "./models/tf_lite_ocr"

    # TODO: Improve
    main(
        camera=args.camera,
        img_path=args.input_image_path,
        tflite_model_location=(lite_model_path + "ocr_model.tflite"),
        new_model=args.new_model,
        epochs=args.epochs
    )


if __name__ == "__main__":
    # TODO: Debugging interface

    # Specify test image input
    img_folder = "./test_images/"
    # img_name = "card.jpeg"
    img_name = "performance.png"

    # Camera input flag
    camera_flag = False

    # TFLite model flags
    new_model_flag = False
    num_epochs = None
    tf_model_path = "./models/ocr"
    tflite_model_path = "./models/tf_lite_ocr/ocr_model.tflite"

    ####################################################################
    # Check for new model flag
    if new_model_flag:
        generate_ocr.generate_ocr_model(tf_model_path, num_epochs)
        tf2tflite(tf_model_path, tflite_model_path)

    # Run!
    main(
        camera=camera_flag,
        img_path=img_folder + img_name,
        tflite_model_location=tflite_model_path,
        new_model=new_model_flag,
        epochs=num_epochs
    )
