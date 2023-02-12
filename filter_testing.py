import cv2
import numpy as np

from main import predict


def gaussian_diff_filter(gray_img, kernel1, kernel2):

    img = gray_img.copy()

    blur1 = cv2.GaussianBlur(img, (kernel1, kernel1), 0)
    blur2 = cv2.GaussianBlur(img, (kernel1, kernel2), 0)
    g_diff = blur2 - blur1

    _, thresh = cv2.threshold(g_diff, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    adapt_thresh = cv2.adaptiveThreshold(g_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)

    cv2.imshow('Threshold Filter', thresh)
    cv2.imshow('Adaptive Threshold Filter', adapt_thresh)
    cv2.imshow(f'Gaussians Diff, Kernels: {kernel1}, {kernel2}', g_diff)

    cv2.waitKey()


def blur_filter(gray_img):

    blur = cv2.bilateralFilter(gray_img, 9, 75, 75)

    _, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    adapt_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3)

    cv2.imshow(f'Threshold Filter, Avg Color: {np.mean(thresh):.2f}', thresh)
    cv2.imshow(f'Adaptive Threshold Filter, Avg Color: {np.mean(adapt_thresh):.2f}', adapt_thresh)

    cv2.waitKey()


def sentence_extract(img):

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply blur and adaptive threshold filter to help finding characters
    # blured = cv2.blur(gray_img, (5, 5), 0)
    # adapt_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)

    # blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # ret, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #
    # # Use findContours to get locations of characters
    # cnts, heirs = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 218])
    upper = np.array([157, 54, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Create horizontal kernel and dilate to connect text characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilate = cv2.dilate(mask, kernel, iterations=5)

    cnts, heirs = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by box location
    bxs = [cv2.boundingRect(c) for c in cnts]
    boxes, hierarchies = zip(*sorted(zip(bxs, heirs[0]), key=lambda b: b, reverse=False))

    result = 255 - cv2.bitwise_and(dilate, mask)

    # Iterate through the list of sorted contours
    for idx, box in enumerate(boxes):
        (x1, y1, w, h) = box
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0))  # BGR

        sentence = result[y1:y1 + h, x1:x1 + w]

        '''
        Begin sentence to character parsing
        '''

        blured = cv2.blur(sentence, (5, 5), 0)
        # blured = cv2.bilateralFilter(sentence, 9, 75, 75)
        adapt_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)

        # Use findContours to get locations of characters
        cnts, heirs = cv2.findContours(adapt_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by box location
        bxs = [cv2.boundingRect(c) for c in cnts]
        boxes, hierarchies = zip(*sorted(zip(bxs, heirs[0]), reverse=False))

        # Iterate through the list of sorted contours
        for i, bx in enumerate(boxes):

            # If a contour has a child, assume it's a letter
            if hierarchies[i][3] != -1:
                (x2, y2, w, h) = bx
                x2 += x1
                y2 += y1
                cv2.rectangle(img, (x2, y2), (x2 + w, y2 + h), (0, 0, 255))  # BGR

    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == "__main__":
    image = cv2.imread("./test_images/this_is_a_test.png", cv2.IMREAD_UNCHANGED)

    sentence_extract(image)

    # for ltrs in snts:
    #     for i, ltr in enumerate(ltrs):
    #         print(f"Image {i + 1}")
    #         plt.imshow(ltr, cmap=plt.cm.binary)
    #         plt.show()
    #         plt.pause(0.2)

    cv2.destroyAllWindows()
