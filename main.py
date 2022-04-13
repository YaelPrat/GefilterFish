import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
carrot_img = cv2.imread("carrot.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
forehead_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    forehead_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        center= (landmarks.part(25).x, landmarks.part(25).y)
        left = (landmarks.part(24).x, landmarks.part(24).y)
        right = (landmarks.part(27).x, landmarks.part(27).y)

        width = int(hypot(left[0] - right[0],
                           left[1] - right[1]) * 1.7)
        height = int(width *1.2 )

        #Position
        top_left = (int(center[0] - width ),
                              int(center[1] - height ))
        bottom_right = (int(center[0] + width ),
                       int(center[1] + height ))

        # Adding the carrot
        carrot = cv2.resize(carrot_img, (width, height))
        carrot_gray = cv2.cvtColor(carrot, cv2.COLOR_BGR2GRAY)
        _, forehead_mask = cv2.threshold(carrot_gray, 25, 255, cv2.THRESH_BINARY_INV)

        area = frame[top_left[1]: top_left[1] + height,
                    top_left[0]: top_left[0] + width]
        place = cv2.bitwise_and(area, area, mask=forehead_mask)
        final = cv2.add(place, carrot)

        frame[top_left[1]: top_left[1] + height,
        top_left[0]: top_left[0] + width] = final

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
