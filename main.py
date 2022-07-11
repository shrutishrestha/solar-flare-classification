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
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
import cv2
import math

#Labels in CSV and Inputs in Fits in a folder
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

#For Confusion Matrix
from sklearn.metrics import confusion_matrix

#Warnings
import warnings
warnings.simplefilter("ignore", Warning)



#-------------------------model-------------------------#
import warnings
from collections import namedtuple
from typing import Optional, Tuple, List, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.hub


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
__all__ = ["GoogLeNet", "googlenet", "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    "googlenet": "https://download.pytorch.org/models/googlenet-1378be20.pth",
}

GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None,
        dropout: float = 0.6,
        dropout_aux: float = 0.7,
        output_num_classes: int = 2,
    ) -> None:
        super().__init__()
        
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of GoogleNet will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, output_num_classes, bias=True)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)

        x = self.fc(x)
        x = F.log_softmax(x)
        

        return x, aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def googlenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> GoogLeNet:
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    The required minimum input size of the model is 15x15.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: True if ``pretrained=True``, else False.
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" not in kwargs:
            kwargs["aux_logits"] = False
        if kwargs["aux_logits"]:
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them"
            )
        original_aux_logits = kwargs["aux_logits"]
        kwargs["aux_logits"] = True
        kwargs["init_weights"] = False
        model = GoogLeNet(**kwargs)
#         state_dict = load_state_dict_from_url(model_urls["googlenet"], progress=progress)
#         model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
        return model

    return GoogLeNet(**kwargs)



#-------------------------model-------------------------#


#Use this if you want to load entiredataset
class MyFitsDataset(Dataset):

    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomRotation(90),
                                            transforms.RandomVerticalFlip(p=1),
                                            transforms.RandomHorizontalFlip(p=1),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.annotations.iloc[index,1].replace(".jpg",".jpeg")))
        image = np.asarray(image)
        h,w,c = image.shape

        image = self.transform(image)
        
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))

        return (image, y_label)

    def __len__(self):
        return len(self.annotations)
    
#Use this if you want to load only flaring instances
class Flare(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, last_drop =0):
        iter_csv = pd.read_csv(csv_file, iterator=True)
        df = pd.concat([chunk[chunk['goes_class'] == 1] for chunk in iter_csv])
        if last_drop:
            df.drop(df.tail(last_drop).index,inplace=True)
        self.annotations = df
        self.root_dir = root_dir


        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomRotation(90),
                                            transforms.RandomVerticalFlip(p=1),
                                            transforms.RandomHorizontalFlip(p=1),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.annotations.iloc[index,1].replace(".jpg",".jpeg")))
        image = np.asarray(image)
        h,w,c = image.shape

        image = self.transform(image)
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))
        return (image, y_label)

    def __len__(self):
        return len(self.annotations)


class NoFlare(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        iter_csv = pd.read_csv(csv_file, iterator=True)
        df = pd.concat([chunk[chunk['goes_class'] == 0] for chunk in iter_csv])
        self.annotations = df
        self.root_dir = root_dir

        self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomRotation(90),
                                            transforms.RandomVerticalFlip(p=1),
                                            transforms.RandomHorizontalFlip(p=1),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
        
        
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.annotations.iloc[index,1].replace(".jpg",".jpeg")))
        image = np.asarray(image)
        h,w,c = image.shape

        image = self.transform(image)
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))

        return (image, y_label)

    def __len__(self):
        return len(self.annotations)

    
def sklearn_Compatible_preds_and_targets(model_prediction_list, model_target_list):

    y_pred_list = []
    preds = []
    target_list = []
    tgts = []
    y_pred_list = [a.tolist() for a in model_prediction_list]
    preds = [item for sublist in y_pred_list for item in sublist]
    target_list = [a.tolist() for a in model_target_list]
    tgts = [item for sublist in target_list for item in sublist]
    return accuracy_score(preds, tgts)

def accuracy_score(prediction, target):

    
    TN, FP, FN, TP = confusion_matrix(target, prediction).ravel()
    print("TP: ", TP, "FP: ", FP, "TN: ", TN, "FN: ", FN)
    #TSS Computation also known as "recall"
    tp_rate = TP / float(TP + FN) if TP > 0 else 0  
    fp_rate = FP / float(FP + TN) if FP > 0 else 0
    TSS = tp_rate - fp_rate
    
    #HSS2 Computation
    N = TN + FP
    P = TP + FN
    HSS = (2 * (TP * TN - FN * FP)) / float((P * (FN + TN) + (TP + FP) * N))
    
    #F0.5 Score Computation
    prec,recall,fscore,_ = precision_recall_fscore_support(target, prediction, average='macro', beta=0.5)

    return TSS, HSS, fscore, TN, FP, FN, TP


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

    # setting the model to train mode
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
        
        scores = model(data)
        
        loss = criterion(scores, targets)
        
        _, p = torch.max(scores,1)

        train_prediction_list.append(p)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()

    checkpoint_file = args["current_checkpoint_file"]+str(epoch)+".pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, checkpoint_file)

    
    
    model.eval()
    val_loss = 0.
    val_tss = 0.
    val_hss = 0.
    val_prediction_list = []
    val_target_list = []
    val_probability_list = []
    with torch.no_grad():
        for d, t in val_loader:

            d = d.to(device=device)
            t = t.to(device=device)
            val_target_list.append(t)
            
            s = model(d)
            
            l = criterion(s, t)

            _, p = torch.max(s,1)

            val_prediction_list.append(p)
            
            val_loss += l.item()

            del d,t,s,l,p
            
            
    scheduler.step(val_loss)
            
    #Epoch Results
    train_loss /= len(train_loader)


    val_loss /= len(val_loader)


    
    train_tss, train_hss, train_fscore, train_TN, train_FP, train_FN, train_TP = sklearn_Compatible_preds_and_targets(train_prediction_list, train_target_list)

    
    val_tss, val_hss, val_fscore, val_TN, val_FP, val_FN, val_TP = sklearn_Compatible_preds_and_targets(val_prediction_list, val_target_list)

    print(f'Epoch: {epoch+1}/{num_epochs}')
    print(f'Training--> loss: {train_loss:.4f}, TSS: {train_tss:.4f}, HSS2: {train_hss:.4f}, F0.5 Score: {train_fscore:.4f} | Val--> loss: {val_loss:.4f}, TSS: {val_tss:.4f} , HSS2: {val_hss:.4f}, F0.5 Score: {val_fscore:.4f} ')
    

    data = [epoch, train_loss, train_tss, train_hss, train_fscore, train_TN, train_FP, train_FN, train_TP, val_loss, val_tss, val_hss, val_fscore, val_TN, val_FP, val_FN, val_TP]
    with open(args["train_val_result_csvpath"],'a') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows([data])

