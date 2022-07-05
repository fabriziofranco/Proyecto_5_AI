import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import utils
import warnings
warnings.filterwarnings("ignore")


class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1, bias=True)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1, bias=True)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=True)
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1, bias=True)
    self.conv_latent = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=True)

    self.conv1Stride = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2, padding=1, bias=True)
    self.conv2Stride = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=2, padding=1, bias=True)
    self.conv3Stride = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=2, padding=1, bias=True)
    self.conv4Stride = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=2, padding=1, bias=True)


  def forward(self, image):
    layer_1 = F.relu(self.conv1(image))
    compress_1 = F.relu(self.conv1Stride(layer_1))
    layer_2 = F.relu(self.conv2(compress_1))
    compress_2 = F.relu(self.conv2Stride(layer_2))
    layer_3 = F.relu(self.conv3(compress_2))
    compress_3 = F.relu(self.conv3Stride(layer_3))
    layer_4 = F.relu(self.conv4(compress_3))
    compress_4 = F.relu(self.conv4Stride(layer_4))
    latent = F.relu(self.conv_latent(compress_4))
    return latent

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.conv_latent = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), stride=1, padding=1, bias=True)
    self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=True)
    self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=1, padding=1, bias=True)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=1, padding=1, bias=True)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(3,3), stride=1, padding=1, bias=True)

  def forward(self, latent):
    from_latent = F.relu(self.conv_latent(latent))
    latent_extended = F.upsample(from_latent, scale_factor=2, mode='nearest')
    layer_1 = F.relu(self.conv1(latent_extended))
    layer_1_extended = F.upsample(layer_1, scale_factor=2, mode='nearest')
    layer_2 = F.relu(self.conv2(layer_1_extended))
    layer_2_extended = F.upsample(layer_2, scale_factor=2, mode='nearest')
    layer_3 = F.relu(self.conv3(layer_2_extended))
    layer_3_extended = F.upsample(layer_3, scale_factor=2, mode='nearest')     
    layer_4 = F.relu(self.conv4(layer_3_extended))
    return F.tanh(layer_4)

class Autoencoder(nn.Module):
   def __init__(self):
      super(Autoencoder, self).__init__()
      self.encoder = Encoder()
      self.decoder = Decoder()

   def forward(self, x):
      latent = self.encoder(x)
      color_prediction = self.decoder(latent)
      return  color_prediction