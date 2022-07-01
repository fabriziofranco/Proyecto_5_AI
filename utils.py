import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from natsort import natsorted
from PIL import Image
from PIL import ImageFile
from skimage import io, color, transform

class LabDataSet(data.Dataset):
    def __init__(self, main_dir, transform, test_size=0.1, height=128, width=128):
        self.AB_scale = 128
        self.main_dir = main_dir
        self.transform = transform
        self.gray_transform = transforms.Compose([transforms.ToTensor()]) 
        self.height = height
        self.width = width
        
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

        train_idx, val_idx = train_test_split(list(range(len(self.total_imgs))), test_size=test_size)
        
        self.train_set = [self.__getitem__(x) for x in train_idx ]
        self.test_set = [self.__getitem__(x) for x in val_idx ]

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        rgb = io.imread(img_loc)
        lab = color.rgb2lab(rgb).astype("float32")
        
        tensor_image_gray = self.gray_transform(lab)
        tensor_image_color = self.transform(lab)

        tensor_image_gray = tensor_image_gray[0,:,:] /100
        tensor_image_color = tensor_image_color[1:,:,:] / self.AB_scale
        return (tensor_image_gray, tensor_image_color, img_loc)