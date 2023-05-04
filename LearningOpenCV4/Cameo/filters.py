import cv2
import numpy as np
import utils


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


class ConvFilter(object):
    def __init__(self, kernel) -> None:
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(ConvFilter):
    def __init__(self) -> None:
        kernel = np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ]
        )
        super().__init__(kernel)


class FindEdgeFilter(ConvFilter):
    def __init__(self) -> None:
        kernel = np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1],
            ]
        )
        super().__init__(kernel)


class BlurFilter(ConvFilter):
    def __init__(self) -> None:
        kernel = np.array(
            [
                [0.04, 0.04, 0.04, 0.04, 0.04],
                [0.04, 0.04, 0.04, 0.04, 0.04],
                [0.04, 0.04, 0.04, 0.04, 0.04],
                [0.04, 0.04, 0.04, 0.04, 0.04],
                [0.04, 0.04, 0.04, 0.04, 0.04],
            ]
        )
        super().__init__(kernel)


class EmbossFilter(ConvFilter):
    def __init__(self) -> None:
        kernel = np.array(
            [
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2],
            ]
        )
        super().__init__(kernel)
