import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
# import time


def segment(image):
    batch_size = image.shape[0]
    markers = []
    img = None

    for batch in range(batch_size):
        img = image[batch].permute(1, 2, 0).cpu().detach().numpy()
        img = np.uint8(img * 255)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dst = cv2.medianBlur(gray, 7)

        gx = cv2.Sobel(dst, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(dst, cv2.CV_32F, 0, 1, ksize=1)

        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        mag = np.uint8(mag * 255 / np.amax(mag))

        ret, thresh = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = 255 - thresh

        kernel_ero = np.ones((2, 2), np.uint8)
        kernel_dil = np.ones((7, 7), np.uint8)

        erosion = cv2.erode(thresh, kernel_ero, iterations=2)
        sure_fg = cv2.dilate(erosion, kernel_dil, iterations=4)
        sure_bg = cv2.dilate(sure_fg, kernel_dil, iterations=7)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)

        markers = markers + 1

        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)

    return markers


def flatten(x, markers):
    max = np.amax(markers)
    markers = np.uint8(markers)
    markers = cv2.resize(markers, (x.shape[3], x.shape[2]))
    markers = markers / np.amax(markers)
    markers = markers * max
    markers = np.uint8(markers)
    markers = markers.flatten()

    # start = time.time()

    x = x.flatten(2)
    x = x[:, :, np.flip(markers.argsort()).copy()]

    # end = time.time()
    # print(end - start)

    return x
