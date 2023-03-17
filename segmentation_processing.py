import cv2
import matplotlib.pyplot as plt
import numpy as np
from functools import cmp_to_key
import tensorflow as tf

from generate_ocr import input_size

# Indexing variables
t = 0
b = 1
x = 0
y = 1

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


def disp_img(image, name, color_map='gray'):
    plt.imshow(image, cmap=color_map)
    plt.title(name)
    plt.show()


def fit(gray_img):
    # Threshold
    thresh = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)[1]

    # Apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((9, 9), np.uint8)
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_ERODE, kernel)

    # Get the largest contour by area
    contours = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get bounding box
    x_val, y_val, w, h = cv2.boundingRect(sorted_cnts[0])
    cropped_img = gray_img[y_val:y_val + h, x_val:x_val + w]

    # Crop result
    return cropped_img


def is_intersecting(box0, box_list):
    for idx, (box1, img) in enumerate(box_list):
        if not (box1[t][x] >= box0[b][x]) and not (box0[t][x] >= box1[b][x]) \
                and not (box1[t][y] >= box0[b][y]) and not (box0[t][y] >= box1[b][y]):
            return idx
    return None


def combine(box0, box1):
    x_vals = box0[t][x], box0[b][x], box1[t][x], box1[b][x]
    y_vals = box0[t][y], box0[b][y], box1[t][y], box1[b][y]

    return (np.min(x_vals), np.min(y_vals)), (np.max(x_vals), np.max(y_vals))


def coords_sort(img_bx1, img_bx2):

    box1 = img_bx1[0]
    box2 = img_bx2[0]

    if box1[t][x] <= box2[t][x]:
        if box1[t][y] >= box2[b][y]:
            return -1
        return 1
    elif box1[b][y] <= box2[t][y]:
        return 1

    return -1


def segment_img(gray_img, debug=False):
    # Check if the image has a black or white background
    if np.mean(gray_img) < 50:
        gray_img = cv2.bitwise_not(gray_img)

    # Crop out any excess white space
    cropped_img = fit(gray_img)

    # Threshold for a image to use for final output images
    thresh0 = cv2.threshold(cropped_img, 127, 255, cv2.THRESH_BINARY)[1]

    # Adaptive threshold + invert for masking
    thresh1 = cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Blend vertically to merge and i's or j's
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    morph1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, rect_kernel)
    thresh2 = cv2.threshold(morph1, 190, 255, cv2.THRESH_BINARY_INV)[1]

    # Thicken things to make boxes a little bigger than the character
    kernel = np.ones((1, 2), np.uint8)
    morph2 = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)

    # Use findContours to find characters
    cnts, heirs = cv2.findContours(morph2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    heirs = heirs[0, :, 3]

    box_img_list = []
    for idx, c in enumerate(cnts):
        if heirs[idx] != -1:

            # Skip areas that cannot be resized properly
            x_val, y_val, w, h = cv2.boundingRect(c)
            if (w / img_resize) > h or (h / img_resize) > w:
                continue

            # Skip blank boxes
            crop = thresh0[y_val:y_val + h, x_val:x_val + w]
            if np.min(crop) == 255 or np.max(crop) == 0:
                continue

            # Check for overlapping boxes and combine them
            box = (x_val, y_val), (x_val + w, y_val + h)
            inter = is_intersecting(box, box_img_list)
            if inter is not None:
                box = combine(box, box_img_list.pop(inter)[0])

            box_image = thresh0[box[t][y]:box[b][y], box[t][x]:box[b][x]]
            # Resize and pad the box
            letter_resize = pad_resize(box_image)
            # Model prefers blurry images
            letter_blur = cv2.blur(letter_resize, (2, 2))
            # Add the box to the list of characters
            img_with_box = (box, letter_blur)

            # Insert the box into a sorted list
            box_img_list.append(img_with_box)

    # Some optional debug info
    if debug:

        disp_img(cropped_img, "cropped_img")
        disp_img(thresh0, "thresh0")
        disp_img(thresh1, "thresh1")
        disp_img(morph1, "mask")
        disp_img(thresh2, "thresh2")
        disp_img(morph2, "morph1")

        ref_img = cropped_img.copy()

        for (box, char) in box_img_list:
            cv2.rectangle(ref_img, box[t], box[b], (0, 0, 0), thickness=1)

        ref_shape = ref_img.shape
        ref_resize = tf.image.resize(
            np.expand_dims(ref_img, axis=2),
            (ref_shape[0] * 10, ref_shape[1] * 10),
            preserve_aspect_ratio=True
        )

        disp_img(ref_resize, "boxes")

    # Sort by boxes from top left to bottom right
    box_img_list.sort(key=cmp_to_key(coords_sort), reverse=True)
    # Discard coordinates, they are no longer needed
    boxes, letters = zip(*box_img_list)
    # Reformat into shape (len(letters), 28, 28)
    letters = np.stack(letters)

    # Add fourth "color" dimension if necessary
    if len(letters.shape) != 4:
        letters = np.expand_dims(letters, axis=3)

    return letters


def display_results(input_data):
    x_pos = [20, 320, 620, 920, 1220]

    # Display each character and its predicted value
    for idx, data_point in enumerate(input_data):

        window_name = f"{idx}"

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, x_pos[idx % 5], 300)
        resize_img = cv2.resize(data_point, (280, 280))
        cv2.imshow(window_name, resize_img)
        cv2.waitKey(1)

        if idx % 5 == 4:
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
            cv2.destroyAllWindows()

    if len(input_data) % 5 != 0:
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    img_list = [
        "this_is_a_test.png",
        "performance.png",
        "tesseract_sample.jpg",
        "card.jpeg",
        "book.png",
    ]
    i = -1

    if i > -1:
        img_list = [img_list[i]]

    for image_name in img_list:
        test_image = cv2.imread("./test_images/" + image_name, cv2.IMREAD_GRAYSCALE)
        chars = segment_img(test_image)

    print("Done! ")
