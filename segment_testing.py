import cv2
import matplotlib.pyplot as plt
import numpy as np
from functools import cmp_to_key
import tensorflow as tf

import segment

# Indexing variables
t = 0
b = 1
x = 0
y = 1


def test_letters_extract(gray_img):
    # TODO: Parse the 'i' and related correctly!
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
    # blured = cv2.boxFilter(clean_img, -1, (5, 5))
    # blured = cv2.bilateralFilter(clean_img, 15, 75, 75)
    # blured = cv2.GaussianBlur(clean_img, (5, 5), 0))
    # blured = cv2.medianBlur(clean_img, 5)
    adapt_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)

    # Sharpen image for later segmentation
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Use findContours to get locations of characters
    cnts, heirs = cv2.findContours(adapt_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by box location (sorted top left to bottom right)
    # bxs = [cv2.boundingRect(c) for c in cnts]
    bxs = [cv2.minAreaRect(c) for c in cnts]
    bxs2 = [cv2.boxPoints(b1) for b1 in bxs]
    bxs3 = [np.int0(b2) for b2 in bxs2]

    contours, boxes, hierarchies = zip(*sorted(zip(cnts, bxs3, heirs[0]), key=lambda bx: bx[1], reverse=False))

    # Iterate through the list of sorted contours
    letters = []
    for idx, box in enumerate(boxes):

        (x_val, y_val, w, h) = box

        # Skip aspect ratios that cannot be scaled to 28x28 properly
        if (w / segment.img_resize) > h or (h / segment.img_resize) > w:
            continue

        # If a contour has a child, assume it's a letter
        if hierarchies[idx][3] != -1:
            # Crop each bounding box
            letter_crop = thresh[y_val:y_val + h, x_val:x_val + w]

            # Skip blank boxes
            if np.min(letter_crop) == 255 or np.max(letter_crop) == 0:
                continue

            # Resize and pad the box
            letter_resize = segment.pad_resize(letter_crop)
            # Model prefers blurry images
            letter_blur = cv2.bilateralFilter(letter_resize, 2, 0, 0)
            # Add the box to the list of characters
            letters.append(letter_blur)

    return np.expand_dims(np.stack(letters), axis=3)


def disp_img(image, name, color_map='gray'):
    plt.imshow(image, cmap=color_map)
    plt.title(name)
    plt.show()


def fit(gray_img):
    # threshold
    thresh = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)[1]

    # apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((9, 9), np.uint8)
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_ERODE, kernel)

    # get the largest contour by area
    contours = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    # bxs = [cv2.minAreaRect(c) for c in sorted_cnts]
    # bxs2 = [cv2.boxPoints(b1) for b1 in bxs]
    # bxs3 = [np.int0(b2) for b2 in bxs2]

    # pt0, pt1, pt2, pt3 = bxs3[0]
    # cropped_img = gray_img[]

    # get bounding box
    x_val, y_val, w, h = cv2.boundingRect(sorted_cnts[0])
    cropped_img = gray_img[y_val:y_val + h, x_val:x_val + w]

    # crop result
    return cropped_img


def is_intersecting(box0, box_list):
    overlap_tolerance = 1.0
    for idx, (box1, img) in enumerate(box_list):
        if not (box1[t][x] >= (box0[b][x] * overlap_tolerance)) \
                and not (box0[t][x] >= (box1[b][x] * overlap_tolerance)) \
                and not (box1[t][y] >= (box0[b][y] * overlap_tolerance)) \
                and not (box0[t][y] >= (box1[b][y] * overlap_tolerance)):
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


def segmentation_test(gray_img, debug=False):
    # Check if the image has a black or white background
    if np.mean(gray_img) < 50:
        gray_img = cv2.bitwise_not(gray_img)

    cropped_img = fit(gray_img)

    thresh0 = cv2.threshold(cropped_img, 127, 255, cv2.THRESH_BINARY)[1]

    thresh1 = cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    morph1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, rect_kernel)

    thresh2 = cv2.threshold(morph1, 190, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = np.ones((1, 2), np.uint8)
    morph2 = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)

    if debug:
        disp_img(cropped_img, "cropped_img")
        disp_img(thresh0, "thresh0")
        disp_img(thresh1, "thresh1")
        disp_img(morph1, "mask")
        disp_img(thresh2, "thresh2")
        disp_img(morph2, "morph1")

    cnts, heirs = cv2.findContours(morph2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    heirs = heirs[0, :, 3]

    box_img_list = []
    ref_img = cropped_img.copy()
    for i, c in enumerate(cnts):
        if heirs[i] != -1:

            # Skip areas that cannot be resized properly
            x_val, y_val, w, h = cv2.boundingRect(c)
            if (w / segment.img_resize) > h or (h / segment.img_resize) > w:
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
            letter_resize = segment.pad_resize(box_image)
            # Model prefers blurry images
            letter_blur = cv2.blur(letter_resize, (2, 2))
            # Add the box to the list of characters
            img_with_box = (box, letter_blur)

            # Insert the box into a sorted list
            box_img_list.append(img_with_box)

    if debug:
        for (box, char) in box_img_list:
            cv2.rectangle(ref_img, box[t], box[b], (0, 0, 0), thickness=1)

        ref_shape = ref_img.shape
        ref_resize = tf.image.resize(
            np.expand_dims(ref_img, axis=2),
            (ref_shape[0] * 10, ref_shape[1] * 10),
            preserve_aspect_ratio=True
        )

        disp_img(ref_resize, "boxes")

    # Sort the boxes from top left to bottom right
    box_img_list.sort(key=cmp_to_key(coords_sort), reverse=True)

    boxes, letters = zip(*box_img_list)

    if debug:
        for i in range(3):
            disp_img(letters[i], f"{i}")

    letters = np.stack(letters)

    if len(letters.shape) == 4:
        return letters
    else:
        return np.expand_dims(letters, axis=3)


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
        # "performance.png",
        # "tesseract_sample.jpg",
        # "card.jpeg",
        # "book.png",
    ]

    for image_name in img_list:
        test_image = cv2.imread("./test_images/" + image_name, cv2.IMREAD_GRAYSCALE)
        chars = segmentation_test(test_image)
        # display_results(chars)
        # input("Press any key to continue... ")

    print("Done! ")
