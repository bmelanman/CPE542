import os.path
import keras
import cv2
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from generateModel_OCR import ocr, input_size, result_arr

model_dir = "./models/ocr"
data_dir = "./datasets"
img_dir_arr = ["letter_c.png", "simple_test_img.png", "tesseract_sample.jpg"]
batch_size = 16


def sort_contours(cnts):
    # Sort contours left to right
    boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, boxes = zip(*sorted(zip(cnts, boxes), key=lambda b: b[1][0], reverse=False))
    return cnts


def disp_result(char_img, result, i):
    char_predictions = [result_arr[x] for x in np.where(result > 0.0)[0]]

    # Get the highest prediction
    index = np.where(result == np.amax(result))
    pred = index[0][0]

    if pred is None:
        return
    elif not len(char_predictions):
        plt.imshow(char_img, cmap=plt.cm.binary)
        plt.title(f"Fig: {i}, Prediction: None")
        plt.show()
        plt.pause(0.5)
        return

    plt.imshow(char_img, cmap=plt.cm.binary)
    plt.title(f"Fig: {i}, Prediction: {result_arr[pred]}")
    plt.show()
    plt.pause(0.5)


def remove_outliers(arr, outlier_const):
    np_arr = np.array(arr)

    upper_quartile = np.percentile(np_arr, 75)
    lower_quartile = np.percentile(np_arr, 25)

    iqr = (upper_quartile - lower_quartile) * outlier_const
    quart_set = (lower_quartile - iqr, upper_quartile + iqr)

    resultList = []
    for ele in np_arr:
        if quart_set[0] <= ele <= quart_set[1]:
            resultList.append(ele)

    return resultList


def pad_resize(orig_image):
    # Arbitrary border width
    border_width = 3
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

    # Normalize image
    img = tf.cast(img, tf.float32) / 255.

    return img


def predict(input_img, ocr_model):
    ref_img = input_img.copy()

    # Filter images for better analysis
    blur = cv2.bilateralFilter(ref_img, 9, 75, 75)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Get a box around each letter
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    contours = sort_contours(contours)

    # Calculate the area of each contour
    contours_area = [cv2.contourArea(cnt) for cnt in contours]

    # TODO: Does this value need to change depending on the image?
    # Remove any statistical outliers
    avg_cnt_area = remove_outliers(contours_area, 3)

    i = 0
    for cnt in contours:

        if i == 30:
            return
        i += 1

        cnt_area = cv2.contourArea(cnt)

        # Get the contour bounds as [x, y, w, h]
        x, y, w, h = cv2.boundingRect(cnt)

        if cnt_area in avg_cnt_area:
            # Crop the image using the data from the contours
            char_img = pad_resize(ref_img[y:y + h, x:x + w])

            # Make sure the cropped image isn't blank
            if np.amax(char_img) == np.amin(char_img):
                return

            # Add 3rd axis for model input
            model_input = np.expand_dims(char_img, axis=0)

            # Predict image
            result = ocr_model.predict(model_input)[0]
            tf.function.__call__()

            # Display the image and prediction for verification
            disp_result(char_img, result, i)


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

    for img in img_dir_arr:
        print(f"Testing {img}...")
        # Load a single PNG containing multiple characters into a CV2 Mat
        test_image = cv2.imread("./test_images/" + img, cv2.IMREAD_GRAYSCALE)
        # Use model to predict the contents of the image
        predict(test_image, ocr_model)
        # return


if __name__ == "__main__":
    main()
