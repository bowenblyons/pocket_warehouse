import cv2
import os
import random

class Camera():
    def capture(self):
        raise NotImplementedError

class SimCamera(Camera):
    def __init__(self, data_dir="test_data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            print("Folder not found")
    def capture(self):
        cat = random.choice([ 'low', 'moderate', 'severe' ])
        cat_path = os.path.join(self.data_dir, cat)
        images = [ image for image in os.listdir(cat_path) if image.lower().endswith(('.jpg'))]
        img = random.choice(images)
        img_path = os.path.join(cat_path, img)
        return img_path
    
