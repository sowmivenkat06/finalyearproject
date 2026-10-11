import os
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset

# pyrefly: ignore [missing-import]
from torchvision import transforms

from config import *

############################################################
# YOLO Object Detection Cropping
############################################################

def yolo_crop(image, yolo_model):
    """
    Run YOLO inference on the image, find the highest-confidence bounding box,
    and crop the original image to that box. If no objects are found, returns the original image.
    """
    # YOLO inference (returns a list of Results objects)
    results = yolo_model.predict(image, verbose=False)
    
    if not results or not len(results[0].boxes):
        return image  # No cells detected, fallback to original
        
    boxes = results[0].boxes
    # Find the index of the box with the highest confidence
    highest_conf_idx = torch.argmax(boxes.conf).item()
    
    # Get xyxy coordinates (x_min, y_min, x_max, y_max)
    x1, y1, x2, y2 = boxes.xyxy[highest_conf_idx].cpu().numpy()
    
    # Add small padding (optional)
    pad = 20
    h_img, w_img = np.array(image).shape[:2]
    
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w_img, int(x2) + pad)
    y2 = min(h_img, int(y2) + pad)
    
    # Crop the image (handling PIL Image or numpy array)
    if isinstance(image, Image.Image):
        return image.crop((x1, y1, x2, y2))
    else:
        # If it's a numpy array
        cropped = image[y1:y2, x1:x2]
        return Image.fromarray(cropped)

############################################################
# CLAHE + Median Filter + Otsu Segmentation
############################################################

def preprocess_image(image, auto_crop=False):

    image = np.array(image)

    # RGB → LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Median Filter
    image = cv2.medianBlur(image,5)

    # Gray Image
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    if auto_crop:
        # Use THRESH_BINARY_INV specifically to find cell contours (cells become white)
        _, contour_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        img_area = image.shape[0] * image.shape[1]
        for c in contours:
            area = cv2.contourArea(c)
            # Filter out tiny noise and background borders
            if area > 100 and area < img_area * 0.90:
                valid_contours.append(c)

        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            pad = 20
            h_img, w_img = image.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            
            # Crop original image and gray image
            image = image[y1:y2, x1:x2]
            gray = gray[y1:y2, x1:x2]

    # Otsu Threshold (Using THRESH_BINARY to match original training logic)
    _,mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )

    # Mask the background (keep cells intact, set background to black)
    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return Image.fromarray(segmented)

############################################################
# Image Transforms
############################################################

transform_224 = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

transform_299 = transforms.Compose([

    transforms.Resize((299,299)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

############################################################
# Dataset Class
############################################################

class CervicalDataset(Dataset):

    def __init__(self,
                 root_dir,
                 transform=None):

        self.transform = transform

        self.image_paths = []

        self.labels = []

        classes = [

            "im_Dyskeratotic",

            "im_Koilocytotic",

            "im_Metaplastic",

            "im_Parabasal",

            "im_Superficial-Intermediate"

        ]

        self.class_names = classes

        for label, cls in enumerate(classes):
            folder = os.path.join(
                root_dir,
                cls,
                "CROPPED"
            )
            if not os.path.exists(folder):
                folder = os.path.join(
                    root_dir,
                    cls
                )

            for image in sorted(os.listdir(folder)):
                if image.lower().endswith(
                    (".bmp",".png",".jpg",".jpeg")
                ):
                    self.image_paths.append(
                        os.path.join(folder,image)
                    )
                    self.labels.append(label)

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self,index):

        image = Image.open(

            self.image_paths[index]

        ).convert("RGB")

        # CLAHE
        image = preprocess_image(image)

        if self.transform:

            image = self.transform(image)

        label = self.labels[index]

        return image,label