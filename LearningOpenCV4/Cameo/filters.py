import cv2
import numpy as np
import utils


class ConvFilter(object):
    def __init__(self, kernel) -> None:
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


def filter(src, dst):
    utils.strokeEdge(src, dst)
