import csv
import numpy as np
import torchvision
import pandas as pd
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import Sampler, WeightedRandomSampler

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
