import cv2
import numpy as np

img=cv2.imread('image.png')


with open ('bounding_box.txt', 'r') as f:
    lines= f.readlines()
    
    for line in lines : 
        l=line.split()
        l2=[int (number) for number in l]
        
        cv2.rectangle(img, (l2[0], l2[1]), (l2[2], l2[3]), (0,0,255),thickness=3)       

cv2.imwrite('hw0_109550198_1.png', img)

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()





