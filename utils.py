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
import shutil
class LabDataSet(data.Dataset):
    def __init__(self, main_dir, transform, train_size=5000, test_size=100, height=128, width=128, seed=0):
        self.AB_scale = 128
        self.main_dir = main_dir
        self.transform = transform
        self.gray_transform = transforms.Compose([transforms.ToTensor()]) 
        self.height = height
        self.width = width
        
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

        train_idx, val_idx = train_test_split(list(range(len(self.total_imgs))), train_size=train_size, test_size=test_size, random_state=seed)
        
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
    test_loss_avg = []
    for epoch in range(Epochs):
        train_loss_avg.append(0)
        test_loss_avg.append(0)
        num_batches = 0
    
        for image_batch, image_batch_r, name_image in train_loader:
            image_batch_r = image_batch_r.to(device)
            
            image_batch = torch.unsqueeze(image_batch, dim=1)
            image_batch = image_batch.to(device)

            image_batch_recon = model(image_batch)
            loss = loss_fn(image_batch_recon, image_batch_r)
            
            model.eval()
            with torch.no_grad():
                iterator = iter(test_loader)
                elements = next(iterator)
                image_batch_valid = torch.unsqueeze(elements[0], dim=1)
                image_batch_valid = image_batch_valid.to(device)
                image_batch_valid_r = model(image_batch_valid)
                loss_valid = loss_fn(image_batch_valid_r, elements[1].to(device))
                test_loss_avg[-1] += loss_valid.item()

            model.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1
            
        train_loss_avg[-1] /= num_batches
        test_loss_avg[-1] /= num_batches
        if epoch%5==0 or epoch == (Epochs - 1):
            print('Epoch [%d / %d] average train error: %f, average test error: %f' % (epoch+1, Epochs, train_loss_avg[-1],  test_loss_avg[-1]))
        if (epoch%20==0 and epoch!=0) or epoch==(Epochs-1):
            plot_batch(device, test_loader, model, height=height, width=width, step = 16)
            model.train()
    return train_loss_avg, test_loss_avg
    


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


def save_results(device, loader, model=None, model_name="model_1", height=256, width=256, is_train=False, batch_size= 32):
    AB_scale = 128
    model.eval()
    with torch.no_grad():
        iteracion = 0
        for image_batch, image_batch_r, name_image in loader:
            image_batch2 = torch.unsqueeze(image_batch, dim=1)
            predictions = []
            resultado = model(image_batch2.to(device))

            for pos in range(len(image_batch)):
                L = (np.asarray(image_batch[pos])) * 100
                A = (resultado[pos][0].detach().cpu().numpy()) *AB_scale
                B = (resultado[pos][1].detach().cpu().numpy()) *AB_scale
                datos = []
                for i in range(L.shape[0]):
                    for j in range(L.shape[1]):
                        datos.append([L[i,j], A[i,j], B[i,j]])

                datos = np.reshape(datos, (height,width,3))
                rgb = color.lab2rgb(datos)
                predictions.append(rgb)


            for pos in range(len(image_batch)):
                figure,axis = plt.subplots(1,3,figsize=(10,10))
                axis[0].set_title("GRAY")
                axis[1].set_title("COLOR")
                axis[2].set_title("PREDICTED")

                axis[0].axis("off")
                axis[1].axis("off")
                axis[2].axis("off")

                L = (np.asarray(image_batch[pos])) * 100
                A = np.asarray(image_batch_r[pos][0]) *AB_scale
                B = np.asarray(image_batch_r[pos][1]) *AB_scale

                datos = []
                for i in range(L.shape[0]):
                    for j in range(L.shape[1]):
                        datos.append([L[i,j], A[i,j], B[i,j]])

                datos = np.reshape(datos, (height,width,3))
                rgb = color.lab2rgb(datos)
                axis[0].imshow(L, cmap='gray')
                axis[1].imshow(rgb)
                axis[2].imshow(predictions[pos])
                if is_train:
                    figure.savefig(f"models/{model_name}/train_results/{(iteracion * batch_size)+pos}_elem.jpg", bbox_inches='tight')
                    plt.close(figure)
                else:
                    figure.savefig(f"models/{model_name}/test_results/{(iteracion * batch_size)+pos}_elem.jpg", bbox_inches='tight')
                    plt.close(figure)
            iteracion += 1
            if iteracion*batch_size >= 2000:
                return "Truncated to 2000 images"

def save_model(device, model, model_name, train_loss_result, test_loss_result, train_loader, test_loader, height=256, width= 256, batch_size=32):
    if os.path.exists(f"models/{model_name}"):
        shutil.rmtree(f"models/{model_name}", ignore_errors=False, onerror=None)
    os.mkdir(os.path.join(f"models/{model_name}"))
    os.mkdir(os.path.join(f"models/{model_name}/train_results"))
    os.mkdir(os.path.join(f"models/{model_name}/test_results"))
    torch.save(model.state_dict(), f"models/{model_name}/model.pt")
    
    textfile = open(f"models/{model_name}/train_loss.txt", "w")
    for element in train_loss_result:
        textfile.write(str(element) + "\n")

    textfile = open(f"models/{model_name}/test_loss.txt", "w")
    for element in test_loss_result:
        textfile.write(str(element) + "\n")

    save_results(device, train_loader,model, model_name, height, width, True, batch_size)
    save_results(device, test_loader,model, model_name, height, width, False, batch_size)