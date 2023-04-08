



import imgaug.augmenters as iaa
import cv2
import numpy as np





image = cv2.imread("image.png")




dx, dy = 50, 50
M = np.float32([[1, 0, dx], [0, 1, dy]])
img_translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))



augmentation = iaa.Sequential([
    iaa.Rotate((-30 ,30))
])
augmented_images = augmentation(images=[image])
img_rotated = augmented_images[0]




img_flipped_horizontal = cv2.flip(image, 1)
#img_flipped_vertical = cv2.flip(image, 0)




scale = 0.5
img_scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)



x1, y1, x2, y2 = 100, 100, 300, 300
img_cropped = image[y1:y2, x1:x2]



cv2.imshow('Translated Image', img_translated)
cv2.imshow('Rotated Image', img_rotated)
cv2.imshow('Flipped Horizontal Image', img_flipped_horizontal)
#cv2.imshow('Flipped Vertical Image', img_flipped_vertical)
cv2.imshow('Scaled Image', img_scaled)
cv2.imshow('Cropped Image', img_cropped)
cv2.waitKey(0)


cv2.imwrite('translated_image.png', img_translated)
cv2.imwrite('rotated_image.png', img_rotated)
cv2.imwrite('flipped_horizontal.png', img_flipped_horizontal)
#cv2.imwrite('flipped_vertical.png', img_flipped_vertical)
cv2.imwrite('scaled_image.png', img_scaled)
cv2.imwrite('cropped_image.png', img_cropped)

cv2.destroyAllWindows()






