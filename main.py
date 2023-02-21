import os.path
import keras
import cv2
import numpy as np
import time

import matplotlib.pyplot as plt

import segment
from generate_ocr import generate_ocr_model, result_arr


def predict(ocr_model, image):

    # Convert the image into a list of characters in image format
    characters = segment.letters_extract(image)

    # Use the model to predict what each character is
    prediction = ocr_model.predict(characters)

    # Display each character and its predicted value
    for idx, letter in enumerate(characters):
        plt.imshow(letter, cmap=plt.cm.binary)
        plt.title(f"Prediction: {result_arr[np.argmax(prediction[idx])]}")
        plt.show()
        plt.pause(0.25)
        if idx > 20:
            break


def camera_input():

    # User input flags
    q_flag = 0
    c_flag = 0

    # Refresh rate in seconds
    refresh_rate = 1

    # Define camera input
    vid = cv2.VideoCapture(0)

    # Display the camera output at the specified framerate
    while True:
        # Refresh rate calculation
        t_end = time.time() + refresh_rate

        # Capture a video frame
        ret, frame = vid.read()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)

        # Trigger display
        cv2.waitKey(1)

        # Loop to wait for next refresh
        while time.time() < t_end:

            # Quit button
            if 0xFF == ord('q'):
                q_flag = 1
                break

            # Capture button
            if 0xFF == ord('c'):
                c_flag = 1
                break

        if q_flag or c_flag:
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    if c_flag:
        return frame

    return 0


def main(new_model=False, epochs=12, camera=False):

    # Directory containing a pre-generated model
    model_dir = "models/ocr"

    # Check if the directory exists
    if not os.path.isdir(model_dir):
        print(f"Error! Please create the directory '{model_dir}'")

    # Check if a model is in the directory, otherwise generate one
    if len(os.listdir(path=model_dir)) == 0 or new_model is True:
        print("Building new model...")
        generate_ocr_model(filepath=model_dir, epochs=epochs)
        print("Model Built!\n")

    # Load the CNN model
    print("Loading model...")
    ocr_model = keras.models.load_model(filepath=model_dir)
    print("Model loaded!\n")

    if camera:
        test_image = camera_input()
        if test_image == 0:
            print("Quit input detected, Goodbye!")
            return 0
        else:
            print("Capture received!")
    else:
        # Image to test
        img = "performance.png"

        # Test the network
        print("Testing model using an image file...")

        # Load an image in grayscale format
        test_image = cv2.imread("./test_images/" + img, cv2.IMREAD_GRAYSCALE)
        print(f"{img} Loaded!")

    # Use model to predict the contents of the image
    predict(ocr_model, test_image)


if __name__ == "__main__":
    main(camera=True)
