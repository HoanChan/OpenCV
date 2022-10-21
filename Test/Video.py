import numpy as np
import cv2
import argparse

#Doc video
video = cv2.VideoCapture(r"d:\slow_traffic.mp4")
#lay frame
ret, frame = video.read()
#xacdinh vi tri dau tien cua window
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)
#vung ROI de do vet
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                   np.array((180., 255., 255.)))

roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while True:
    ret, frame = video.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), 225, 2)
        cv2.imshow('img', img)

        k = cv2.waitKey(30) & 0xff
        if k == ord("q"):
            break
    else:
        break
cv2.destroyAllWindows()
