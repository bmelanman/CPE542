import cv2
import matplotlib.pyplot as plt
import numpy as np

import segment


def test_letters_extract(gray_img):
    # NOTE: The input image should already be grayscale!

    # Check if the image has a black or white background
    if np.mean(gray_img) < 50:
        gray_img = cv2.bitwise_not(gray_img)

    vertical_hist = gray_img.shape[0] - np.sum(gray_img, axis=0, keepdims=True) / 255
    plt.plot(vertical_hist[0])
    plt.imshow(gray_img)
    plt.show()

    # Reduce image noise (w/o None, dst is 4 and the last two become 21, added None->)
    # https://stackoverflow.com/questions/59424253/weird-behavior-of-cv2-fastnlmeansdenoising
    clean_img = cv2.fastNlMeansDenoising(gray_img, None, 4, 7, 21)

    # Apply blur and adaptive threshold filter to help finding characters
    blured = cv2.blur(clean_img, None, (5, 5), 0)
    adapt_thresh = cv2.adaptiveThreshold(blured, None, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)

    # Sharpen image for later segmentation
    ret, thresh = cv2.threshold(gray_img, None, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Use findContours to get locations of characters
    cnts, heirs = cv2.findContours(adapt_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by box location (sorted top left to bottom right)
    # bxs = [cv2.boundingRect(c) for c in cnts]
    bxs = [cv2.minAreaRect(c) for c in cnts]
    bxs2 = [cv2.boxPoints(b) for b in bxs]
    bxs3 = [np.int0(b2) for b2 in bxs2]

    contours, boxes, hierarchies = zip(*sorted(zip(cnts, bxs3, heirs[0]), key=lambda b: b[1], reverse=False))

    # Iterate through the list of sorted contours
    letters = []
    for idx, box in enumerate(boxes):

        (x, y, w, h) = box

        # Skip aspect ratios that cannot be scaled to 28x28 properly
        if (w / segment.img_resize) > h or (h / segment.img_resize) > w:
            continue

        # If a contour has a child, assume it's a letter
        if hierarchies[idx][3] != -1:
            # Crop each bounding box
            letter_crop = thresh[y:y + h, x:x + w]

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


def fit(gray_img):
    # threshold
    thresh = cv2.threshold(gray_img, 190, 255, cv2.THRESH_BINARY)[1]

    # apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((9, 9), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    largest_cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # get bounding box
    x, y, w, h = cv2.boundingRect(largest_cnt)

    # draw filled contour on black background
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [largest_cnt], -1, (255, 255, 255), cv2.FILLED)

    # crop result
    return gray_img[y:y + h, x:x + w]


def segmentation_test(gray_img):
    plt.imshow(gray_img, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.show()

    cropped_img = fit(gray_img)
    plt.imshow(cropped_img, cmap='gray')
    plt.title("fit")
    plt.show()

    blured = cv2.medianBlur(cropped_img, 13)
    plt.imshow(blured, cmap='gray')
    plt.title("blured")
    plt.show()

    adapt_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    plt.imshow(adapt_thresh, cmap='gray')
    plt.title("adapt_thresh")
    plt.show()

    cnts, heirs = cv2.findContours(adapt_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(cropped_img, cmap='gray')
    plt.title("cnts")
    plt.show()

    print('')


if __name__ == "__main__":
    image_name = "card.jpeg"
    # image_name = "performance.png"

    test_image = cv2.imread("./test_images/" + image_name, cv2.IMREAD_GRAYSCALE)

    segmentation_test(test_image)
