import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


class Load(Dataset):

    def __init__(self, img_dir, label_dir, transform = None):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open (label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_centre = float(parts[1])
                    y_centre = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    # converting YOLO format bb -> COCO format bb
                    x1 = (x_centre - w/2) * width
                    y1 = (y_centre - h/2) * height
                    x2 = (x_centre + w/2) * width
                    y2 = (y_centre + h/2) * height
                    boxes.append([x1,y1,x2,y2])
                    labels.append([class_id])
        