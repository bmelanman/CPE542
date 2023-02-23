import os.path
from pathlib import Path

import keras
import cv2
import numpy as np

import segment
from generate_ocr import generate_ocr_model, result_arr


def predict(ocr_model, image):
    # Convert the image into a list of characters in image format
    characters = segment.letters_extract(image)

    # Use the model to predict what each character is
    prediction = ocr_model.predict(characters)

    # Display each character and its predicted value
    for idx, letter in enumerate(characters):

        x_pos = [20, 320, 620, 920, 1220]

        window_name = f"Image {idx}, Prediction: {result_arr[np.argmax(prediction[idx])]}"

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, x_pos[idx % 5], 280)
        resize_img = cv2.resize(letter, (280, 280))
        cv2.imshow(window_name, resize_img)
        cv2.waitKey(1)

        if idx % 5 == 4:
            if cv2.waitKey(0) == ord('q'):
                exit(0)
            cv2.destroyAllWindows()

    cv2.waitKey(0)

    cv2.destroyAllWindows()


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


def main(camera=False, new_model=False, epochs=12):
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
        if test_image.any():
            print("Capture received!")
        else:
            print("Quit input detected, Goodbye!")
            return 0
    else:
        # Image to test
        img_folder = "./test_images/"
        # img = "card.jpeg"
        # img = "tesseract_sample.jpg"
        img = "performance.png"

        if not Path(img_folder + img).is_file():
            print("File not found! Make sure the file name and extension are spelt correctly. ")
            exit(1)

        # Test the network
        print("Testing model using an image file...")

        # Load an image in grayscale format
        test_image = cv2.imread(img_folder + img, cv2.IMREAD_GRAYSCALE)
        print(f"{img} Loaded!")

    # Use model to predict the contents of the image
    predict(ocr_model, test_image)


if __name__ == "__main__":
    main()
