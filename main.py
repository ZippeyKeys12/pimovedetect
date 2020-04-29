from typing import Tuple

import cv2
import imutils
import numpy as np
import serial
from imutils.video import VideoStream

from zippcv.detectors import SSDObjectDetector

DEBUG = False
SERIAL = True

SENSITIVITY = 10

CONF_CUTOFF = .6
DIST_CUTOFF = 36  # Sqaured to save sqrt cost

RECT_COLOR = np.array((0, 255, 255), dtype=np.float)
LINE_COLOR = np.array((255, 255, 255), dtype=np.float)
AXES_COLOR = np.array((0, 0, 255), dtype=np.float)

RECTS = []


def sign(x: float) -> int:
    return int(x > 0) - int(x < 0)


def get_center(rect) -> Tuple[int, int]:
    return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)


def get_magnitude(rect) -> float:
    (x, y) = get_center(rect)
    return (x - CENTER[0])**2 + (y - CENTER[1])**2


def log(frame, label, confidence, rect, color, W):
    if confidence < CONF_CUTOFF:
        return

    RECTS.append(rect)


ssdd = SSDObjectDetector(log, includes=["person"])

vs = VideoStream(usePiCamera=True).start()
frame = vs.read()
frame = imutils.resize(frame, width=600)

HEIGHT, WIDTH, _ = frame.shape
CENTER = (WIDTH // 2, HEIGHT // 2)

if DEBUG:
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 24, (WIDTH, HEIGHT))

if SERIAL:
    ser = serial.Serial("/dev/ttyACM0", 9600)

try:
    while(True):
        frame = vs.read()

        frame = imutils.resize(frame, width=600)

        ssdd(frame)

        if DEBUG:
            # Draw axes
            cv2.line(frame, (0, HEIGHT//2), (WIDTH, HEIGHT//2), AXES_COLOR)
            cv2.line(frame, (WIDTH//2, 0), (WIDTH//2, HEIGHT), AXES_COLOR)

            # Draw boundign boxes and ray from origin to center of box
            for rect in RECTS:
                cv2.rectangle(frame, tuple(rect[:2]), tuple(
                    rect[-2:]), RECT_COLOR, 2)
                cv2.arrowedLine(frame, CENTER, get_center(rect), LINE_COLOR, 1)

            RECTS.clear()

            # Display
            out.write(frame)
            cv2.imshow("Image", frame)

            # Use the escape key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        if len(RECTS) > 0:
            # Target coordinate
            target = get_center(sorted(RECTS, key=get_magnitude)[-1])

            yaw = target[0] - CENTER[0]
            pitch = target[1] - CENTER[1]

            # Allows for imprecision
            if abs(yaw) < DIST_CUTOFF:
                yaw = 0
            else:
                yaw = sign(yaw)

            if abs(pitch) < DIST_CUTOFF:
                pitch = 0
            else:
                pitch = sign(pitch)

            fire = yaw == 0 and pitch == 0

            if SERIAL:
                ser.write(f'<{-yaw},{-pitch},{fire}>\n')

except KeyboardInterrupt:
    pass


if DEBUG:
    out.release()
    cv2.destroyAllWindows()
vs.stop()
