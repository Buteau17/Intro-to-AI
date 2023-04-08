


import cv2 
import numpy as np

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    if not ret:  
        break

    diff = cv2.absdiff(frame1, frame2)

    b, g, r = cv2.split(diff)
    zeros = np.zeros(diff.shape[:2], dtype="uint8")
    diff = cv2.merge([zeros, g, zeros])

    result = np.hstack((frame2, diff))

    cv2.imshow("Difference Frame", result)

    if cv2.waitKey(30) == 1:
        cv2.imwrite("hw0_109550198_2.png", result)
        break

cap.release() 

cv2.destroyAllWindows()
