import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    #raise NotImplementedError("To be implemented")
    dataset=[]
    
    for image_name in os.listdir(os.path.join(data_path,"car")):
        
        img = cv2.imread(os.path.join(data_path,"car", image_name))
        img = cv2.resize(img, (36, 16))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tuple=(img, 1) 
        dataset.append(tuple)

    for image_name in os.listdir(os.path.join(data_path,"non-car")):
        img = cv2.imread(os.path.join(data_path,"non-car", image_name))
        img = cv2.resize(img, (36, 16))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        tuple=(img, 0) 
        dataset.append(tuple)
    # End your code (Part 1)
    return dataset
