import argparse
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import generate_ocr
import segmentation_processing
from generate_ocr import result_arr, input_size


def tf2tflite(load_filepath="./models/ocr", save_filepath="./models/tflite_ocr/"):
    # Load regular TF model
    model = tf.keras.models.load_model(load_filepath)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    lite_model = converter.convert()

    # Save model
    with open(save_filepath + "ocr_model.tflite", 'wb') as f:
        f.write(lite_model)

    return


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


def display_results(input_data, prediction_data, pred_min: float):
    # Remove any data with a prediction of less than the given confidence minimum
    filtered_list = [x for x in zip(input_data, prediction_data) if np.max(x[1]) > pred_min]

    # Display each character and its predicted value
    for idx, data_point in enumerate(filtered_list):

        x_pos = [20, 320, 620, 920, 1220]

        pred = data_point[1]
        window_name = f"{result_arr[np.argmax(pred)]} - {np.max(pred) * 100:0.2f}% - {idx}, "

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, x_pos[idx % 5], 450)
        resize_img = cv2.resize(data_point[0], (280, 280))
        cv2.imshow(window_name, resize_img)
        cv2.waitKey(1)

        if idx % 5 == 4:
            if cv2.waitKey(0) == ord('q'):
                exit(0)
            cv2.destroyAllWindows()

    if len(filtered_list) % 5 != 0:
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_results(input_data, prediction_data, pred_min: float):
    # Remove any data with a prediction of less than the given confidence minimum
    filtered_list = [x for x in zip(input_data, prediction_data) if np.max(x[1]) > pred_min]

    pred_str = ""
    for _, prediction in filtered_list:
        pred_str += result_arr[np.argmax(prediction)]

    print(f"\nThe given image is predicted to contain the following: \n"
          f"\'{pred_str}\'\n"
          f"\n"
          f"Done!\n")

    return 0


def main(camera, img_path, tflite_model_location, pred_min: float, debug=False):
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
    characters = segmentation_processing.segment_img(test_image, debug=debug)

    # Optional use of ArmNN Library
    arm_nn_delegate = None
    if is_raspberrypi():
        print("Raspberry Pi Detected!\n")

        lib_path = "/lib/armnn/libarmnnDelegate.so"

        if Path(lib_path).is_file():
            print(f"ArmNN delegate library found!")
            print(f"Library location: {lib_path}")

            os.environ["GLOG_minloglevel"] = "3"

            arm_nn_delegate = tf.lite.experimental.load_delegate(
                library=lib_path,
                options={
                    "backends": "CpuAcc,GpuAcc,CpuRef",
                    "logging-severity": "info"
                }
            )
        else:
            print(f"Could not find ArmNN Delegate library at \'{lib_path}\'")
            print("The ArmNN library will not be used\n")

    # Load TFLite model and set the input size to the number of characters
    interpreter = tf.lite.Interpreter(
        model_path=tflite_model_location,
        experimental_delegates=[arm_nn_delegate]
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
    if debug:
        display_results(characters, prediction, pred_min)
    else:
        print_results(characters, prediction, pred_min)

    return 0


def user_cli():
    err_flag = 0

    # Disable logging from TensorFlow while not in debug mode
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    print("---------------------------------------- PORC-Y ---------------------------------------\n"
          "Hello, welcome to PORC-Y: The Python-based Optical Recognition of Characters, partiallY\n")

    parser = argparse.ArgumentParser()

    parser.add_argument(    # Camera options
        "-c", "--camera",
        type=str,
        default="True",
        help="Enable webcam input (False by default)"
    )
    parser.add_argument(    # Image options
        "-i", "--input-image-path",
        type=str,
        default=None,
        help="A path to an image to process"
    )
    parser.add_argument(    # Model options
        "-m", "--model-path",
        type=str,
        default=None,
        help="A path to a model.tflite file"
    )

    args = parser.parse_args()

    # Convert the camera option to boolean
    camera_bool = args.camera.lower() in ("yes", "true", "t", "1")

    # Check for valid configurations
    if args.input_image_path is None and camera_bool is False:
        print("Please provide an image file or enable webcam input!")
        err_flag = 1
    if args.model_path is None:
        print("Please provide a model.tflite file!")
        err_flag = 1

    if err_flag == 1:
        exit(err_flag)
    # User Prompt
    print(f" - Camera input is set to:  {camera_bool}")
    print(f" - Image input is set to:   {not camera_bool}")
    print(f" - Image path is set to:    {args.input_image_path}")
    print(f" - Model Path is set to:    {args.model_path}\n")
    input("Press any key to continue...\n")

    main(
        camera=camera_bool,
        img_path=args.input_image_path,
        tflite_model_location=args.model_path,
        pred_min=0.5
    )

    return 0


def debug_interface():
    # Calculate apx. execution time
    t0 = time.time()

    # Specify test image input
    img_folder = "./test_images/"
    img_list = [
        "performance.png",
        "this_is_a_test.png",
        "card.jpeg",
        "book.png",
    ]
    # Specify test image index, or use -1 to test all images
    img_idx = 1

    # Minimum prediction confidence
    min_conf = 0.40

    # Camera input flag
    camera_flag = False

    # TFLite model flags
    create_new_model_flag = False
    load_new_model_flag = True
    num_epochs = None
    tf_model_path = "./models/ocr"
    tflite_model_path = "models/tflite_ocr/ocr_model.tflite"

    # Print debug info and images
    debug = True

    ####################################################################
    # Check for new model flag
    if create_new_model_flag:
        generate_ocr.generate_ocr_model(tf_model_path, num_epochs)
    if create_new_model_flag or load_new_model_flag:
        tf2tflite(tf_model_path, tflite_model_path)

    if img_idx >= 0:
        img_list = [img_list[img_idx]]

    for img_name in img_list:
        # Run!
        main(
            camera=camera_flag,
            img_path=img_folder + img_name,
            tflite_model_location=tflite_model_path,
            pred_min=min_conf,
            debug=debug
        )

    print(f"Processing Time: {time.time() - t0:.2f}\n")

    return 0


if __name__ == "__main__":
    if os.environ.get("DEBUG") == '1':
        print("Using debugging interface...")
        debug_interface()
    else:
        user_cli()

    exit(0)