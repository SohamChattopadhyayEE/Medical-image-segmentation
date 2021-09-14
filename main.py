import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split


import util.evaluation.*
import util.predict_image.*
import util.dice_coeff.*
import util.data_view.*
from data.dataset import BasicDataset
from models import R2UNet.R2U_Net as seg_model

import warnings
warnings.filterwarnings("ignore")

model_name = ["UNet", "AtttentionUNet", "R2U_Net", "R2AttentionUNet"] 
segmentation_model = model_name[2]

dir_img_train = '/content/drive/MyDrive/Brain Segmentation/train/Inputs'
dir_mask_train = '/content/drive/MyDrive/Brain Segmentation/train/Outputs'
dir_img_val = '/content/drive/MyDrive/Brain Segmentation/val/Inputs'
dir_mask_val = '/content/drive/MyDrive/Brain Segmentation/val/Outputs'
dir_checkpoint = '/content/drive/MyDrive/Brain Segmentation'


batch_size = 30
num_epoch = 200
num_channels = 3
num_classes = 1
lr = 0.0001
img_scale = 1
mask_threshold = 0.5


dataset_train = BasicDataset(dir_img_train, dir_mask_train, img_scale, mask_suffix='_mask')
dataset_val = BasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix='_mask')

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)#, num_workers=8, pin_memory=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)#, num_workers=8, pin_memory=True, drop_last=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if segmentation_model == "UNet":
    net = seg_model(n_channels = num_channels, n_classes = num_classes)
else:
    net = seg_model(img_ch=num_channels,output_ch=num_classes)




net = net.to(device=device)
optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=lr)
if net.n_classes > 1:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()
criterion = DiceCoeff()

iou = IoU(num_classes = 2)

for epoch in range(num_epoch):
  net.train()
  train_loss = 0.0
  miou = total = 0
  print('Epoch: ', epoch)
  for itr, data in enumerate(train_loader):
    #print('Training data: ', itr)
    optimizer.zero_grad()

    image, mask = data['image'], data['mask']
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    image = image.to(device=device, dtype = torch.float32) 
    original_mask = mask.to(device=device, dtype = mask_type)

    pred_mask = net(image)
    #loss = criterion(pred_mask, original_mask)
    loss = dice_coeff(pred_mask, original_mask)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    total += mask.size(0)
    iou.add(pred_mask, original_mask)
    _, miou_temp =  iou.value()
    miou += miou_temp
  print('Train loss: ', train_loss/len(dataset_train))
  print('mIoU: ', miou*100/total)
  fn = '/content/drive/MyDrive/Brain Segmentation/train/Inputs/TCGA_DU_6405_19851005_58.tif'
  fn_mask = '/content/drive/MyDrive/Brain Segmentation/train/Outputs/TCGA_DU_6405_19851005_58_mask.tif'
  img = Image.open(fn)
  img_mask = Image.open(fn_mask)
  mask = predict_img(net=net,
                      full_img=img,
                      scale_factor=img_scale,
                      out_threshold=mask_threshold,
                      device=device)
  plot_img_and_mask(img_mask, mask)


  net.eval()
  with torch.no_grad():
    val_loss = 0.0
    for iter, val_data in enumerate(val_loader):
      #print('Validation data: ', iter)
      image, mask = val_data['image'], val_data['mask']
      mask_type = torch.float32 if net.n_classes == 1 else torch.long
      image = image.to(device=device, dtype = torch.float32) 
      original_mask = mask.to(device=device, dtype = mask_type)

      pred_mask = net(image)
      #loss = criterion(pred_mask, original_mask)  
      loss = dice_coeff(pred_mask, original_mask) 
      val_loss += loss.item()
    print('validation loss: ', val_loss/len(dataset_val)) 
  fn = '/content/drive/MyDrive/Brain Segmentation/test/Inputs/TCGA_HT_A61B_19991127_69.tif'
  fn_mask = '/content/drive/MyDrive/Brain Segmentation/test/Outputs/TCGA_HT_A61B_19991127_69_mask.tif'
  img = Image.open(fn)
  img_mask = Image.open(fn_mask)
  mask = predict_img(net=net,
                      full_img=img,
                      scale_factor=img_scale,
                      out_threshold=mask_threshold,
                      device=device)
  plot_img_and_mask(img_mask, mask)
    
