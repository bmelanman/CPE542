import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from generate_ocr import input_size


def check_dark_background(input_img):
    # image grayscale and filtering
    image = cv2.imread(input_img)

    # another section code to try out
    # gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,9,3)
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    avg_color_row = np.average(input_img, axis=0)
    avg_color = np.average(avg_color_row, axis=0)

    if avg_color < 50:
        return True

    return False


def pad_resize(orig_image):
    # Arbitrary border width
    border_width = 2
    # New image size based off border width
    new_img_size = input_size - (border_width * 2)

    # Add 3rd dimension for resize and model input
    expand = np.expand_dims(orig_image, axis=2)
    # Invert because resize pads with 0's
    invert = np.invert(expand)

    # Resize image to new size
    resize = tf.image.resize(invert, (new_img_size, new_img_size), preserve_aspect_ratio=True)
    # Pad image to get to final size
    pad = tf.image.pad_to_bounding_box(resize, border_width, border_width, input_size, input_size)

    # Normalize image
    normalize = tf.cast(pad, tf.float32) / 255.

    return normalize


def letters_extract(gray_img):

    # Check if the image has a black or white background
    #if check_dark_background(gray_img):
    #    gray_img = cv2.bitwise_not(gray_img)

    # adjust kernal sizes to rid of smaller contour edges(play around with this to get optimal results)
    kernal1 = 15
    kernal2 = 31
    blur1 = cv2.GaussianBlur(image, (kernal1, kernal1), 0)
    blur2 = cv2.GaussianBlur(image, (kernal1, kernal2), 0)
    finalblur = blur1 - blur2
    #to ensure the output is good enough to distinguish text from background
    cv2.imshow('Difference of Gaussians', finalblur)
    thresh = cv2.adaptiveThreshold(finalblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)
    cv2.imshow('after thresh', thresh)

    # Apply blur and threshold filter to help finding characters
    #blur = cv2.bilateralFilter(gray_img, 9, 75, 75)
    #ret, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Use findContours to get locations of characters
    cnts, heirs = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by box location
    bxs = [cv2.boundingRect(c) for c in cnts]
    contours, boxes, hierarchy = zip(*sorted(zip(cnts, bxs, heirs[0]), key=lambda b: b[1], reverse=False))

    # Iterate through the list of sorted contours
    letters = []
    for idx, contour in enumerate(contours):

        # If a contour has a child, assume it's a letter
        if hierarchy[idx][3] != -1:
            (x, y, w, h) = boxes[idx]
            letter_crop = thresh[y:y + h, x:x + w]
            letter_resize = pad_resize(letter_crop)
            letters.append(letter_resize)

    return np.stack(letters)


if __name__ == "__main__":

    print("Loading image...")
    o = cv2.imread("./test_images/performance.png", cv2.IMREAD_UNCHANGED)
    print("Imaged loaded!")

    print("Extracting letters...")
    ltrs = letters_extract(o)
    print("Letters extracted, displaying...")

    for i, ltr in enumerate(ltrs):
        print(f"Image {i + 1}")
        plt.imshow(ltr, cmap=plt.cm.binary)
        plt.show()
        plt.pause(0.2)
