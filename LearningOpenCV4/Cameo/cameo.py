import cv2
from manager import WindowManager, CaptureManager


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager("cameo", self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True
        )

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is not None:
                pass
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


if __name__ == "__main__":
    Cameo().run()
