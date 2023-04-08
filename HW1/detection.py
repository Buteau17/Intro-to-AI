import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")
    # Load data from the specified file path
    with open(data_path, "r") as f:
        nums = int(f.readline().strip())
        coordinates = []
        for i in range(nums):
            num_list = f.readline().strip().split(' ') # ['101', '102']

            x = []
            for num in num_list:
                x.append(int(num))
            
            coordinates.append(tuple(x))

    # Open a video file for processing
    cap = cv2.VideoCapture('data/detect/video.gif')
    idx=0
    with open('ML_Models_pred.txt', 'a') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for i in range (int(nums)):
                cropped_frame = crop(coordinates[i][0],coordinates[i][1],coordinates[i][2],coordinates[i][3],coordinates[i][4],coordinates[i][5],coordinates[i][6],coordinates[i][7],frame)
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                cropped_frame = cv2.resize(cropped_frame, (36, 16), interpolation=cv2.INTER_AREA)  # Resize the cropped frame to a fixed size
                
                # print(cropped_frame.shape)
                cropped_frame=np.reshape(cropped_frame,(1,-1))
                # print(cropped_frame.shape)
                if clf.classify(cropped_frame) == 1:
                    f.write('1 ')
                    green_color = (0, 255, 0) # BGR
                    pts = np.array([[coordinates[i][0], coordinates[i][1]], [coordinates[i][2], coordinates[i][3]], [coordinates[i][6], coordinates[i][7]], [coordinates[i][4], coordinates[i][5]]], np.int32)
                    pts = pts.reshape((4, 1, 2))# Reshape the coordinates array
                    cv2.polylines(frame, [pts], True, green_color, 1)# Draw a polygon on the frame
                else:
                    f.write('0 ')
            cv2.imshow('Image', frame)
            f.write('\n')
            if idx==0:
                cv2.imwrite('Image.png',frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                index=1
            if cv2.waitKey(30) == 13:
                break
    
    
    cv2.destroyAllWindows()
    cap.release()

    
    # End your code (Part 4)
