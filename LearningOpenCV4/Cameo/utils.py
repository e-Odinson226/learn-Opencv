import scipy
import cv2
import numpy as np


def strokeEdge(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        bluredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(bluredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInversed = 1 / 255 * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInversed
    cv2.merge(channels, dst)

    # cv2.imshow("normalizedInversed", normalizedInversed)
    # cv2.imshow("dst", dst)
