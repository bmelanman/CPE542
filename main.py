import os.path
import keras
import cv2
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from generateModel_OCR import ocr, input_size, result_arr

model_dir = "./models/ocr"
data_dir = "./datasets"
test_img_dir = "./test_images/simple_test_img.png"
batch_size = 16


def sort_contours(cnts):
    # Sort contours left to right
    boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, boxes = zip(*sorted(zip(cnts, boxes), key=lambda b: b[1][0], reverse=False))
    return cnts


def pad_resize(orig_image):
    # Arbitrary border width
    border_width = 4
    # New image size based off border width
    new_img_size = input_size - (border_width * 2)

    # Add 3rd dimension for resize and model input
    img = np.expand_dims(orig_image, axis=2)
    # Invert because resize pads with 0's
    img = np.invert(img)

    # Resize image to new size
    img = tf.image.resize(img, (new_img_size, new_img_size), preserve_aspect_ratio=True)
    # Pad image to get to final size
    img = tf.image.pad_to_bounding_box(img, border_width, border_width, input_size, input_size)

    return img
    # orig_size = orig_image.shape
    # padding_size = max(orig_size) - min(orig_size)
    #
    # # Add padding onto the shorter axis to make a square image
    # if orig_size[1] > orig_size[0]:
    #     char_img = cv2.copyMakeBorder(orig_image, padding_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
    # else:
    #     char_img = cv2.copyMakeBorder(orig_image, 0, 0, padding_size, 0, cv2.BORDER_CONSTANT, value=255)

    # tf.image.resize(image, (img_size, img_size)


def test(img, model):
    # Add 3rd axis for model input
    test_image = np.expand_dims(img, axis=0)

    # Process image
    result = model.predict(test_image)

    # Make sure the picture isn't blank
    if np.amax(test_image) != np.amin(test_image):

        # Get the highest prediction
        index = np.where(result == np.amax(result))
        return index[1][0]


def predict(input_img, ocr_model):
    img = input_img.copy()

    # Filter images for better analysis
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # Get a box around each letter
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    avg_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        avg_area += (w * h)

    # TODO: Does this value need to change depending on the image?
    margin = 0.9
    avg_area /= len(contours)
    max_area = (1 + margin) * avg_area
    min_area = (1 - margin) * avg_area
    i = 0
    for cnt in contours:
        # Get the contour bounds as [x, y, w, h]
        x, y, w, h = cv2.boundingRect(cnt)
        print(max_area, w * h, min_area)

        # If the box is "character-sized" then run it through the model
        if max_area > w * h > min_area:

            char_img = pad_resize(img[y:y + h, x:x + w])
            pred = test(char_img, ocr_model)

            if pred is None:
                continue

            plt.imshow(char_img, cmap=plt.cm.binary)
            plt.title(f"Fig {i}, Prediction: {result_arr[pred]}, Size: {w * h}")
            plt.show()
            plt.pause(0.5)

        i += 1


def main(new_model=False, epochs=3):
    if not os.path.isdir(model_dir):
        print(f"Error! Please create the directory '{model_dir}'")

    if len(os.listdir(path=model_dir)) == 0 or new_model is True:
        print("Building new model...")
        ocr(filepath=model_dir, epochs=epochs)
        print("Model Built!\n")

    print("Loading model...")
    ocr_model = keras.models.load_model(filepath=model_dir)
    print("Model loaded!\n")

    # Test the network
    print("Testing model...")
    # Load a single PNG containing multiple characters into a CV2 Mat
    test_image = cv2.imread(test_img_dir, cv2.IMREAD_GRAYSCALE)
    # Use model to predict the contents of the image
    predict(test_image, ocr_model)


if __name__ == "__main__":
    main()
    cv2.waitKey(10)
    cv2.destroyAllWindows()
