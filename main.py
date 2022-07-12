import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.models.vgg import model_urls
# from models.models import  googlenet
from astropy.io import fits as fits
import torch.nn as nn 
import torch.nn.functional as F
import csv
import torch.optim as optim
import torchvision
from dataset import MyFitsDataset
import cv2
import math

#Labels in CSV and Inputs in Fits in a folder
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

#Warnings
import warnings
warnings.simplefilter("ignore", Warning)

from model import googlenet
from dataloader import FitsDataset
from metrics import sklearn_Compatible_preds_and_targets, accuracy_score


job_id = os.getenv('SLURM_JOB_ID')
args = {}


args["model"] = "googlenet"
snapshot_main_dir = os.path.join("/scratch", str(job_id), "snapshots")
snapshot_model_dir = os.path.join(snapshot_main_dir, args["model"])

if not os.path.isdir(snapshot_main_dir):
    os.mkdir(snapshot_main_dir)
    
if not os.path.isdir(snapshot_model_dir):
    os.mkdir(snapshot_model_dir)
    
args["current_checkpoint_file"] = snapshot_model_dir +  "/current_"
        
train_val_result_main_dir = os.path.join("/scratch", str(job_id), "results")
train_val_result_model_dir = os.path.join(snapshot_main_dir, args["model"])

if not os.path.isdir(train_val_result_main_dir):
    os.mkdir(train_val_result_main_dir)
    
if not os.path.isdir(train_val_result_model_dir):
    os.mkdir(train_val_result_model_dir)
    
args["train_val_result_csvpath"] = train_val_result_model_dir +  "/result.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# # Load Dataj
partition1_path = '/scratch/shruti_dataset/dataset/512_CENTERCROP/class_gt_M_Partition1.csv'
partition2_path = '/scratch/shruti_dataset/dataset/512_CENTERCROP/class_gt_M_Partition2.csv'
partition3_path = '/scratch/shruti_dataset/dataset/512_CENTERCROP/class_gt_M_Partition3.csv'
partition4_path = '/scratch/shruti_dataset/dataset/512_CENTERCROP/class_gt_M_Partition4.csv'

partition1_folder = '/scratch/shruti_dataset/dataset/512_CENTERCROP/SELFMASKED_34_PARTITIONS/PARTITION1/'
partition2_folder = '/scratch/shruti_dataset/dataset/512_CENTERCROP/SELFMASKED_34_PARTITIONS/PARTITION2/'
partition3_folder = '/scratch/shruti_dataset/dataset/512_CENTERCROP/SELFMASKED_34_PARTITIONS/PARTITION3/'
partition4_folder = '/scratch/shruti_dataset/dataset/512_CENTERCROP/SELFMASKED_34_PARTITIONS/PARTITION4/'



partition1_noflare = NoFlare(csv_file = partition1_path, 
                             root_dir = partition1_folder,
                             transform = True)
partition1_flare = Flare(csv_file = partition1_path, 
                             root_dir = partition1_folder,
                             transform = True)

partition1_flare187 = Flare(csv_file = partition1_path, 
                             root_dir = partition1_folder,
                             transform = True, last_drop =110)

partition2_noflare = NoFlare(csv_file = partition2_path, 
                             root_dir = partition2_folder,
                             transform = True)
partition2_flare = Flare(csv_file = partition2_path, 
                             root_dir = partition2_folder,
                             transform = True)


partition3_noflare = NoFlare(csv_file = partition3_path, 
                             root_dir = partition3_folder,
                             transform = True)
partition3_flare = Flare(csv_file = partition3_path, 
                             root_dir = partition3_folder,
                             transform = True)

partition4_noflare = NoFlare(csv_file = partition4_path, 
                             root_dir = partition4_folder,
                             transform = True)
partition4_flare = Flare(csv_file = partition4_path, 
                             root_dir = partition4_folder,
                             transform = True)
partition4_flare181 = Flare(csv_file = partition4_path, 
                             root_dir = partition4_folder,
                             transform = True, last_drop =54)


train_set = ConcatDataset([partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,
                           partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,
                           partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,
                           partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,
                           partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare,partition2_flare, partition2_noflare,
                           
                          
                          partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,             partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,
partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare,partition4_flare, partition4_flare181, partition4_noflare
                          ])

val_set = MyFitsDataset(csv_file = partition1_path, 
                             root_dir = partition1_folder)

batch_size = 32

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4, shuffle = True)

val_loader = DataLoader(dataset=val_set, batch_size=1, num_workers=4, shuffle=False)


# Hyperparameters
in_channel = 3
learning_rate = 0.0001
num_epochs = 100

# Initialize network
model = googlenet(pretrained = True).to(device)

# Loss and optimizer
criterion = nn.NLLLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

#Train_Model
# Training Network
print("Training in Progress..")

print("1,2,3 train, 4 val")
for epoch in range(num_epochs):

    # trainig starts
    model.train()
    train_loss = 0
    train_tss = 0.
    train_hss = 0.
    train_prediction_list = []
    train_target_list = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        
        data = data.to(device=device)
        targets = targets.to(device=device)
        train_target_list.append(targets)
        
        # for every mini-batch during the training phase, we should explicitly set the gradients to zero before starting to do 
        # backpropragation as PyTorch accumulates the gradients on subsequent backward passes, if we do not do this, gradient will accumulate from the 
        # first iteration, leading to wrong gradients
        optimizer.zero_grad() 
        
        #our googlenet model
        output_train = model(data) 
        
        # NLLLoss to calculate how many magnetograms have been predicted as flaring class
        loss_train = criterion(output_train, targets) 
        
         #taking out probability
        _, p_train = torch.max(scores_train,1)
        train_prediction_list.append(p_train) 
        
        #upgrading gradients
        loss_train.backward() 
        
        #After computing the gradients for all tensors, optimizer.step() now makes the optimizer iterate over all parameters (tensors) to update them 
        #with their internally stored grad
        optimizer.step() 
        
        #collecting loss
        train_loss += loss_train.item() 

    # saving checkpoints
    checkpoint_file = args["current_checkpoint_file"]+str(epoch)+".pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, checkpoint_file)
    
    # validation starts
    model.eval()
    val_loss = 0.
    val_tss = 0.
    val_hss = 0.
    val_prediction_list = []
    val_target_list = []
    val_probability_list = []
    with torch.no_grad():
        for data, target in val_loader:

            data = data.to(device=device)
            target = target.to(device=device)
            val_target_list.append(target)
            
            output_val = model(data) 
            loss_val = criterion(output_val, target)

            _, p_val = torch.max(output_val,1)
            val_prediction_list.append(p_val)
            val_loss += loss_val.item()
            
            
    scheduler.step(val_loss)
            
    #Epoch Results
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    train_tss, train_hss, train_fscore, train_TN, train_FP, train_FN, train_TP = sklearn_Compatible_preds_and_targets(train_prediction_list, train_target_list)
    val_tss, val_hss, val_fscore, val_TN, val_FP, val_FN, val_TP = sklearn_Compatible_preds_and_targets(val_prediction_list, val_target_list)

    data = [epoch, train_loss, train_tss, train_hss, train_fscore, train_TN, train_FP, train_FN, train_TP, val_loss, val_tss, val_hss, val_fscore, val_TN, val_FP, val_FN, val_TP]
    with open(args["train_val_result_csvpath"],'a') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows([data])

