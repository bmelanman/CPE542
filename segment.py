import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from generateModel_OCR import input_size

def check_dark_background(input_img):

    avg_color_row = np.average(input_img, axis=0)
    avg_color = np.average(avg_color_row, axis=0)

    if avg_color < 50:
        return True

    return False


def sort_contours(contours, hierarchy):
    # Sort contours left to right
    boxes = [cv2.boundingRect(c) for c in contours]
    cnts, _, heirs = zip(*sorted(zip(contours, boxes, hierarchy[0]), key=lambda b: b[1], reverse=False))
    return cnts, heirs


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


def letters_extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if check_dark_background(gray):
        gray = cv2.bitwise_not(gray)

    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    ret, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # RETR_TREE, RETR_LIST, RETR_EXTERNAL, and RETR_CCOMP
    # [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = sort_contours(contours, hierarchy)

    letters = []
    for idx, contour in enumerate(contours):

        if idx >= len(hierarchy):
            break

        if hierarchy[idx][3] != -1:
            (x, y, w, h) = cv2.boundingRect(contour)
            letter_crop = gray[y:y + h, x:x + w]
            letter_resize = pad_resize(letter_crop)
            letters.append(letter_resize)

    return np.stack(letters)


if __name__ == "__main__":
    # img_dir_arr = ["simple_test_img.png", "letter_c.png", "tesseract_sample.jpg", "ocr_test.png"]
    #
    # for image in img_dir_arr:
    #     o = cv2.imread(f"test_images/{image}", cv2.IMREAD_UNCHANGED)
    #     letters = letters_extract(o)

    image = "simple_test_img.png"

    o = cv2.imread(f"test_images/{image}", cv2.IMREAD_UNCHANGED)
    ltrs = letters_extract(o)

    for ltr in ltrs:
        plt.imshow(ltr)
        plt.show()
        plt.pause(0.2)
