import cv2
from manager import WindowManager, CaptureManager
import filters
import depth


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager("cameo", self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True
        )
        self._embossedFilter = filters.EmbossFilter()

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:

                filters.strokeEdge(frame, frame)
                self._embossedFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keypress):
        if keypress == 32:
            self._captureManager.writeImage("screenshot.jpg")
        elif keypress == 9:
            if self._captureManager.isWritingVideo:
                self._captureManager.stopWritingVideo()
            elif not self._captureManager.isWritingVideo:
                self._captureManager.startWriteVideo()
        elif keypress == 27:
            self._windowManager.destorWindow()


class CameoDepth(Cameo):
    def __init__(self):
        self._windowManager = WindowManager("cameo", self.onKeypress)
        device = cv2.CAP_OPENNI2
        # device = cv2.CAP_OPENNI2_ASUS
        self._captureManager = CaptureManager(
            cv2.VideoCapture(device), self._windowManager, True
        )
        self._embossedFilter = filters.EmbossFilter()

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            self._captureManager.channel = cv2.CAP_OPENNI_DISPARITY_MAP
            disparityMap = self._captureManager.frame

            self._captureManager.channel = cv2.CAP_OPENNI_VALID_DEPTH_MASK
            validDepthMask = self._captureManager.frame

            self._captureManager.channel = cv2.CAP_OPENNI_BGR_IMAGE
            frame = self._captureManager.frame

            if frame is None:
                self._captureManager.channel = cv2.CAP_OPENNI_IR_IMAGE
                frame = self._captureManager.frame

            if frame is not None:
                mask = depth.createMedianMask(disparityMap, validDepthMask)
                frame[mask == 0] = 0
                if self._captureManager.channel == cv2.CAP_OPENNI_BGR_IMAGE:
                    filters.strokeEdge(frame, frame)
                    self._embossedFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()


if __name__ == "__main__":
    # Cameo().run()
    CameoDepth().run()
