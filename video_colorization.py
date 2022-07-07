from datetime import timedelta
import cv2
import os
from PIL import Image
from natsort import natsorted
import shutil
from moviepy.editor import *
from moviepy.video import *
import utils
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import model_1
import numpy as np
from skimage import io, color, transform
import matplotlib.pyplot as plt


def create_directory(directory):
    # make a folder by the name of the video file
    if not os.path.isdir(directory):
        os.mkdir(directory)

def video_to_images(video_path, data_dst, filename):
    # filename, _ = os.path.splitext(video_path)

    create_directory(data_dst+filename)
    create_directory(f"{data_dst}{filename}/tmp")
    
    count=1
    vid = video_path
    vidcap = cv2.VideoCapture(vid)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(data_dst+filename+"/tmp/"+str(count)+".jpg", image) # Save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 1/30 # Change this number to 1 for each 1 second
    
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 20)
        success = getFrame(sec)


def images_to_video(data_path, data_dst, video_name, frames=30):
    img_array = []
    number_of_frames = frames
    create_directory(data_dst)
    for img in natsorted(os.listdir(data_path)):
        img = cv2.imread(data_path+img)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter(data_dst+video_name,cv2.VideoWriter_fourcc(*'MPEG'), 30, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def resize_and_save_img(src, destination_path, color="RGB"):
    original_img = Image.open(src).convert(color).resize((256,256))
    original_img.save(destination_path)

def resize_all_images_and_colorized(data_path, data_dst, color="RGB"):
    create_directory(data_dst)
    for img in natsorted(os.listdir(data_path)):
        resize_and_save_img(data_path + img, data_dst + img, color)


def split_videos(data_path, filename):
    font_size = 22
    stroke_width = 2
    font_color = 'white'
    font_stroke = 'black'

    clip_color_original = VideoFileClip(data_path + filename + "_color.avi")
    original_txt = TextClip("Original", fontsize = font_size, color = font_color, stroke_color= font_stroke, stroke_width=stroke_width) 
    original_txt = original_txt.set_pos('top').set_duration(clip_color_original.duration)
    clip_color_original = CompositeVideoClip([clip_color_original, original_txt]) 

    clip_color_predicted = VideoFileClip(data_path + filename + "_predicted.avi")
    prediceted_txt = TextClip("Predicted", fontsize = font_size, color = font_color, stroke_color= font_stroke, stroke_width=stroke_width) 
    prediceted_txt = prediceted_txt.set_pos('top').set_duration(clip_color_predicted.duration)
    clip_color_predicted = CompositeVideoClip([clip_color_predicted, prediceted_txt]) 

    clip_gray = VideoFileClip(data_path + filename + "_gray.avi")
    grayscale_txt = TextClip("Grayscale", fontsize = font_size, color = font_color, stroke_color= font_stroke, stroke_width=stroke_width)
    grayscale_txt = grayscale_txt.set_pos('top').set_duration(clip_gray.duration)
    clip_gray = CompositeVideoClip([clip_gray, grayscale_txt]) 
    
    
    
    result = clips_array([[clip_gray, clip_color_original, clip_color_predicted]])
    os.remove(data_path + filename + "_color.avi")
    os.remove(data_path + filename + "_gray.avi")
    os.remove(data_path + filename + "_predicted.avi")
    

    #save video
    result.write_videofile(f"{data_path}{filename}_resultado.mp4")


def colorize_grays(model_name,video_path, filename, height=256, width=256, duration=15, frames=30):
    img_transform = transforms.Compose([transforms.ToTensor()]) 
    dataset = utils.LabDataSet(f'videos/{filename}/color',
                img_transform, train_size=int(duration*frames/2), test_size=int(duration*frames/2), height=height, width=width, seed=3)

    model = model_1.Autoencoder()
    model.load_state_dict(torch.load(f'models/{model_name}/model.pt', map_location=torch.device('cpu') ))
    model.eval()

    train_loader = torch.utils.data.DataLoader(dataset=dataset.train_set, batch_size=int(duration*frames/2), shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset.test_set, batch_size=int(duration*frames/2), shuffle=False)
    device = torch.device('cpu')
    os.mkdir(f'videos/{filename}/predictions')

    step = 1
    AB_scale = 128
    model.eval()
    with torch.no_grad():
        for loader in [test_loader, train_loader]:
            iterator = iter(loader)
            elements = next(iterator)

            image_batch2 = torch.unsqueeze(elements[0], dim=1)
            predictions = []
            names = []
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

            longitud = len(f"videos/{filename}/color")
            names =[]

            for i in range(len(predictions)):
                names.append(elements[2][i][longitud+1:])

            for i in range(len(predictions)):
                Image.fromarray((predictions[i] * 255 ).astype(np.uint8)).convert("RGB").save(f"videos/{filename}/predictions/{names[i]}")


def colorize_video(model_name,video_path, filename, height=256, width=256, duration=15, frames=30):
    video_to_images(video_path, data_dst="videos/", filename=filename)
    resize_all_images_and_colorized(data_path=f"videos/{filename}/tmp/",data_dst=f"videos/{filename}/color/",color="RGB")
    resize_all_images_and_colorized(data_path=f"videos/{filename}/tmp/",data_dst=f"videos/{filename}/gray/",color="L")
    colorize_grays(model_name,video_path, filename, height=256, width=256, duration=duration, frames=frames)
    images_to_video(data_path=f"videos/{filename}/color/", data_dst=f"videos/{filename}/", video_name=filename+"_color.avi", frames = frames) #generar video a color resize
    images_to_video(data_path=f"videos/{filename}/gray/", data_dst=f"videos/{filename}/", video_name=filename+"_gray.avi", frames = frames) #generar video en blanco y negro resize
    images_to_video(data_path=f"videos/{filename}/predictions/", data_dst=f"videos/{filename}/", video_name=filename+"_predicted.avi", frames = frames) #generar video en blanco y negro resize
    shutil.rmtree(f'videos/{filename}/tmp', ignore_errors=False, onerror=None)
    split_videos(data_path=f"videos/{filename}/", filename=filename)
    shutil.rmtree(f'videos/{filename}/color', ignore_errors=False, onerror=None)
    shutil.rmtree(f'videos/{filename}/gray', ignore_errors=False, onerror=None)