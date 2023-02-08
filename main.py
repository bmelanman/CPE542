import os.path
import keras
import cv2
import numpy as np

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
        plt.pause(0.2)


def main(new_model=False, epochs=12):

    # Directory containing a pre-generated model
    model_dir = "./models/ocr"

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

    # Images to test
    img_dir_arr = ["tesseract_sample.jpg"]

    # Test the network
    print("Testing model...")
    for img in img_dir_arr:
        print(f"Testing {img}...")

        # Load an image in grayscale format
        test_image = cv2.imread("./test_images/" + img, cv2.IMREAD_GRAYSCALE)

        # Use model to predict the contents of the image
        predict(ocr_model, test_image)


if __name__ == "__main__":
    main(True, epochs=3)
