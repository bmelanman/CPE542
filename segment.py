import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from generate_ocr import input_size


def check_dark_background(input_img):
    # image grayscale and filtering
    # image = cv2.imread(input_img)

    # another section code to try out
    # gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,9,3)
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    return np.mean(input_img) < 50


def pad_resize(orig_image):
    # Arbitrary border width
    border_width = 2
    # New image size based off border width
    img_resize = input_size - (border_width * 2)

    # Add 3rd dimension for resize and model input
    expand = np.expand_dims(orig_image, axis=2)
    # Invert because resize pads with 0's
    invert = np.invert(expand)

    # Resize image to new size
    resize = tf.image.resize(invert, (img_resize, img_resize), preserve_aspect_ratio=True)

    # Padding size to center the image
    img_shape = resize.shape

    if img_shape[1] != img_resize:
        x_pad = int((img_resize - img_shape[1]) / 2) + border_width
        y_pad = border_width
    else:
        x_pad = border_width
        y_pad = int((img_resize - img_shape[0]) / 2) + border_width

    # Pad image to get to final size
    pad = tf.image.pad_to_bounding_box(resize, y_pad, x_pad, input_size, input_size)

    # Normalize image
    normalize = tf.cast(pad, tf.float32) / 255.

    return np.array(normalize)


def letters_extract(gray_img):

    # The input image should already be grayscale!

    # Check if the image has a black or white background
    if check_dark_background(gray_img):
        gray_img = cv2.bitwise_not(gray_img)

    # Apply blur and adaptive threshold filter to help finding characters
    blured = cv2.blur(gray_img, (5, 5), 0)
    adapt_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)

    # Sharpen image for later segmentation
    ret, thresh = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Use findContours to get locations of characters
    cnts, heirs = cv2.findContours(adapt_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by box location
    bxs = [cv2.boundingRect(c) for c in cnts]
    _, boxes, hierarchies = zip(*sorted(zip(cnts, bxs, heirs[0]), key=lambda b: b[1], reverse=False))

    # Iterate through the list of sorted contours
    letters = []
    for idx, box in enumerate(boxes):

        # If a contour has a child, assume it's a letter
        if hierarchies[idx][3] != -1:
            (x, y, w, h) = box
            letter_crop = thresh[y:y + h, x:x + w]
            letter_resize = pad_resize(letter_crop)
            letter_blur = cv2.bilateralFilter(letter_resize, 2, 0, 0)
            letters.append(letter_blur)

    return np.stack(letters)


if __name__ == "__main__":

    print("Loading image...")
    o = cv2.imread("./test_images/performance.png", cv2.IMREAD_GRAYSCALE)
    print("Imaged loaded!")

    print("Extracting letters...")
    ltrs = letters_extract(o)
    print("Letters extracted, displaying...")

    for i, ltr in enumerate(ltrs):
        print(f"Image {i + 1}")
        plt.imshow(ltr, cmap=plt.cm.binary)
        plt.show()
        plt.pause(0.2)
