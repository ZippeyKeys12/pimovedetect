from typing import Callable, List, Optional, Tuple

import imutils
import numpy as np
from imutils.video import FPS

import cv2


class SSDObjectDetector:
    def __init__(self, drawer:
                 Optional[Callable[[np.array, str, float,
                                    Tuple[int, int, int, int],
                                    Tuple[int, int, int], int], None]] = None,
                 confidence: float = .2,
                 includes: Optional[List[str]] = None,
                 excludes: Optional[List[str]] = None):
        self.drawer = drawer
        self.confidence = confidence
        self.includes = includes
        self.excludes = excludes

        model = "mobilenet/{}".format

        self.LABELS = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow",
                       "diningtable", "dog", "horse", "motorbike", "person",
                       "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        if includes is not None and excludes is not None:
            raise ValueError(
                "'includes' and 'excludes' can't both not be 'None'")

        np.random.seed(42)
        self.COLORS = np.random.uniform(0, 255, size=(len(self.LABELS), 3))

        self.net = cv2.dnn.readNetFromCaffe(
            model("MobileNetSSD_deploy.prototxt.txt"),
            model("MobileNetSSD_deploy.caffemodel"))

    def __call__(self, image: np.array):
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 0.007843, (300, 300), 127.5)

        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence:
                idx = int(detections[0, 0, i, 1])

                if (self.includes is not None and
                        self.LABELS[idx] not in self.includes):
                    continue

                if (self.excludes is not None and
                        self.LABELS[idx] in self.excludes):
                    continue

                box = detections[0, 0, i, 3:7] * np.array(2*[W, H])
                (startX, startY, endX, endY) = box.astype("int")

                info = (self.LABELS[idx], confidence,
                        (startX, startY, endX, endY), self.COLORS[idx])

                if self.drawer is not None:
                    self.drawer(image, *info, W)

                # yield info


def main():
    def draw(frame, label, confidence, rect, color, W):
        color = tuple(map(int, color))

        cv2.rectangle(frame, tuple(rect[:2]), tuple(rect[-2:]), color, 2)

        y = rect[1] - 10 if rect[1] - 10 > 10 else rect[1] + 10
        cv2.putText(frame, "{}: {:.4f}".format(label, confidence * 100),
                    (rect[0], y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vs = cv2.VideoCapture("videos/race.mp4")

    fps = FPS().start()

    object_detector = SSDObjectDetector(draw)

    for _ in range(675):
        vs.grab()

    while True:
        (_, frame) = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=448)

        object_detector(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.release()


if __name__ == "__main__":
    main()
