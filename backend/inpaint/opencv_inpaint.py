import cv2

class OpenCVInpaint:

    def __init__(self):
        pass

    def inpaint(self, frame, mask):
        return cv2.inpaint(frame, mask, 3, cv2.INTER_LINEAR)

    def __call__(self, frames, mask):
        comp = []
        for frame in frames:
            comp.append(self.inpaint(frame, mask))
        return comp