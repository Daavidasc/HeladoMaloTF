import mss
import pyautogui
import cv2
import numpy as np
import time
import multiprocessing

import Template.settings as settings

class ScreenCaptureAgent:
    def __init__(self) -> None:
        self.capture_process = None
        self.fps = None
        self.img = None

        self.w, self.h = pyautogui.size()
        print(f"Screen Resolution: {self.w}x{self.h}")

        self.monitor = {
            'top': settings.COMP_VIZ_TOP_LEFT[1],
            'left': settings.COMP_VIZ_TOP_LEFT[0],
            'width': settings.COMP_VIZ_BOTTOM_RIGHT[0],
            'height': settings.COMP_VIZ_BOTTOM_RIGHT[1]
        }

    def capture_screen(self):
        with mss.mss() as sct:
            while True:
                self.img = sct.grab(self.monitor)
                self.img = np.array(self.img)

                if settings.ENABLE_PREVIEW:
                    preview = cv2.resize(self.img, (0,0), fx= 0.5, fy=0.5)
                    cv2.imshow('Computer Vision', preview)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    agent = ScreenCaptureAgent()
    agent.capture_screen()