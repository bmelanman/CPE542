import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from generate_ocr import input_size
from segment import pad_resize


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


if __name__ == "__main__":
    gray = cv2.imread("./test_images/performance.png", cv2.IMREAD_GRAYSCALE)

    gaussian_diff_filter(gray, 15, 31)
    blur_filter(gray)
    cv2.destroyAllWindows()
