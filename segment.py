import cv2
import numpy as np


def check_dark_background(input_img):

    avg_color_row = np.average(input_img, axis=0)
    avg_color = np.average(avg_color_row, axis=0)

    if avg_color < 50:
        return True

    return False


def letters_extract(image_file):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if check_dark_background(gray):
        gray = cv2.bitwise_not(gray)

    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    ret, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # RETR_TREE, RETR_LIST, RETR_EXTERNAL and RETR_CCOMP.
    # [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    letters = []
    for idx, contour in enumerate(contours):

        if hierarchy[0][i][3] != -1:
            (x, y, w, h) = cv2.boundingRect(contour)
            letter_crop = gray[y:y + h, x:x + w]
            letters.append(letter_crop)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0))     # BGR

        i += 1

    cv2.imshow("gray", gray)
    cv2.imshow("thresh", thresh)
    cv2.imshow(f"Num contours: {len(letters)}", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return letters


if __name__ == "__main__":
    img_dir_arr = ["simple_test_img.png", "letter_c.png", "tesseract_sample.jpg", "ocr_test.png"]

    for image in img_dir_arr:
        letters_extract(f"test_images/{image}")
