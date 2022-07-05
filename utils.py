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
import sys

class LabDataSet(data.Dataset):
    def __init__(self, main_dir, transform, train_size=5000, test_size=100, height=128, width=128):
        self.AB_scale = 128
        self.main_dir = main_dir
        self.transform = transform
        self.gray_transform = transforms.Compose([transforms.ToTensor()]) 
        self.height = height
        self.width = width
        
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

        train_idx, val_idx = train_test_split(list(range(len(self.total_imgs))), train_size=train_size, test_size=test_size)
        
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

def train(device, model, train_loader, test_loader, Epochs, loss_fn, optimizer, height, width):
    train_loss_avg = [] 
    for epoch in range(Epochs):
      train_loss_avg.append(0)
      num_batches = 0
    
      for image_batch, image_batch_r, name_image in train_loader:
          image_batch_r = image_batch_r.to(device)
          
          image_batch = torch.unsqueeze(image_batch, dim=1)
          image_batch = image_batch.to(device)

          image_batch_recon = model(image_batch)
          loss = loss_fn(image_batch_recon, image_batch_r)
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          train_loss_avg[-1] += loss.item()
          num_batches += 1
          
      train_loss_avg[-1] /= num_batches
      if epoch%5==0:
        print('Epoch [%d / %d] average error: %f' % (epoch+1, Epochs, train_loss_avg[-1]))
      if epoch%20==0 and epoch!=0:
        plot_batch(device, test_loader, model, height=height, width=width, step = 16)
        model.train()
    return train_loss_avg


def plot_batch(device, test_loader, model = None ,height=256, width=256, step=32):
    AB_scale = 128
    model.eval()
    with torch.no_grad():
        iterator = iter(test_loader)
        elements = next(iterator)

        image_batch2 = torch.unsqueeze(elements[0], dim=1)
        predictions = []
        
        resultado = model(image_batch2.to(device))

        for pos in range(len(elements[0])):
            L = (np.asarray(elements[0][pos])) * 100
            A = (resultado[pos][0].detach().cpu().numpy()) *AB_scale
            B = (resultado[pos][1].detach().cpu().numpy()) *AB_scale
            datos = []
            for i in range(L.shape[0]):
                for j in range(L.shape[1]):
                    datos.append([L[i,j], A[i,j], B[i,j]])

            datos = np.reshape(datos, (height,width,3))
            rgb = color.lab2rgb(datos)
            predictions.append(rgb)


        for pos in range(len(elements[0])):
            if pos%step!=0:
                continue
            figure,axis = plt.subplots(1,3,figsize=(10,10))
            axis[0].set_title("GRAY")
            axis[1].set_title("COLOR")
            axis[2].set_title("PREDICTED")

            axis[0].axis("off")
            axis[1].axis("off")
            axis[2].axis("off")

            L = (np.asarray(elements[0][pos])) * 100
            A = np.asarray(elements[1][pos][0]) *AB_scale
            B = np.asarray(elements[1][pos][1]) *AB_scale

            datos = []
            for i in range(L.shape[0]):
                for j in range(L.shape[1]):
                    datos.append([L[i,j], A[i,j], B[i,j]])

            datos = np.reshape(datos, (height,width,3))
            rgb = color.lab2rgb(datos)
            axis[0].imshow(L, cmap='gray')
            axis[1].imshow(rgb)
            axis[2].imshow(predictions[pos])
            plt.show()
