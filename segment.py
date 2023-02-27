import cv2
import numpy as np
import tensorflow as tf

from generate_ocr import input_size

import matplotlib.pyplot as plt

# Drawing contours
DRAW_ALL_CNTS = -1
# Arbitrary border width
border_width = 2
# New image size based off border width
img_resize = input_size - (border_width * 2)


def pad_resize(orig_image):

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

    if x_pad < 1 or y_pad < 1:
        print(
            f"ERR: PAD LESS THAN ONE!\n"
            f"x_pad: {x_pad}\n"
            f"x_pad: {img_shape[1]}\n"
            f"x_pad: {y_pad}\n"
            f"x_pad: {img_shape[0]}\n"
        )
        if x_pad < 1:
            x_pad = 1
        else:
            y_pad = 1

    # Pad image to get to final size
    pad = tf.image.pad_to_bounding_box(resize, y_pad, x_pad, input_size, input_size)

    # Normalize image
    normalize = tf.cast(pad, tf.float32) / 255.

    return np.array(normalize)


def letters_extract(gray_img):
    # NOTE: The input image should already be grayscale!

    # Check if the image has a black or white background
    if np.mean(gray_img) < 50:
        gray_img = cv2.bitwise_not(gray_img)

    vertical_hist = gray_img.shape[0] - np.sum(gray_img, axis=0, keepdims=True) / 255
    plt.plot(vertical_hist[0])
    plt.imshow(gray_img)
    plt.show()

    # Reduce image noise
    clean_img = cv2.fastNlMeansDenoising(gray_img, 4, 7, 21)

    # Apply blur and adaptive threshold filter to help finding characters
    blured = cv2.blur(clean_img, (5, 5), 0)
    adapt_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)

    # Sharpen image for later segmentation
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Use findContours to get locations of characters
    cnts, heirs = cv2.findContours(adapt_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by box location (sorted top left to bottom right)
    bxs = [cv2.boundingRect(c) for c in cnts]
    contours, boxes, hierarchies = zip(*sorted(zip(cnts, bxs, heirs[0]), key=lambda b: b[1], reverse=False))

    # Iterate through the list of sorted contours
    letters = []
    for idx, box in enumerate(boxes):

        (x, y, w, h) = box

        # Skip aspect ratios that cannot be scaled to 28x28 properly
        if (w / img_resize) > h or (h / img_resize) > w:
            continue

        # If a contour has a child, assume it's a letter
        if hierarchies[idx][3] != -1:
            # Crop each bounding box
            letter_crop = thresh[y:y + h, x:x + w]

            # Skip blank boxes
            if np.min(letter_crop) == 255 or np.max(letter_crop) == 0:
                continue

            # Resize and pad the box
            letter_resize = pad_resize(letter_crop)
            # Model prefers blurry images
            letter_blur = cv2.bilateralFilter(letter_resize, 2, 0, 0)
            # Add the box to the list of characters
            letters.append(letter_blur)

    return np.expand_dims(np.stack(letters), axis=3)
