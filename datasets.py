import os
import glob
import numpy as np
import nibabel as nib
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from melTransforms import *

class Scan_Dataset_Segm(Dataset):
  def __init__(self, data_dir, transform=False):
    self.transform = transform
    self.img_list = sorted(glob.glob(os.path.join(data_dir,'img*.nii.gz')))
    self.msk_list = sorted(glob.glob(os.path.join(data_dir,'msk*.nii.gz')))

  def __len__(self):
    """defines the size of the dataset (equal to the length of the data_list)"""
    return len(self.img_list)

  def __getitem__(self, idx):
    """ensures each item in data_list is randomly and uniquely assigned an index (idx) so it can be loaded"""

    if torch.is_tensor(idx):
      idx = idx.tolist()

    # loading image
    image_name = self.img_list[idx]
    image = nib.load(image_name).get_fdata()

    # loading mask
    mask_name = self.msk_list[idx]
    mask = nib.load(mask_name).get_fdata()
    mask = np.expand_dims(mask, axis=2)

    # make sample
    sample = {'image': image, 'mask': mask}

    # apply transforms
    if self.transform:
      sample = self.transform(sample)

    return sample
  
class Scan_DataModule_Segm(pl.LightningDataModule):
  def __init__(self, config):
    super().__init__()
    self.train_data_dir   = config['train_data_dir']
    self.val_data_dir     = config['val_data_dir']
    # self.test_data_dir    = config['test_data_dir']
    self.batch_size       = config['batch_size']

    # self.train_transforms = transforms.Compose([Random_Rotate_Seg(0.1), ToTensor_Seg()])
    # self.val_transforms   = transforms.Compose([ToTensor_Seg()])

    self.train_transforms = transforms.Compose([Random_Rotate_Seg(0.1), Random_Horizontal_Flip_Seg(0.1), Random_Vertical_Flip_Seg(0.1),  ToTensor_Seg(), Color_Jitter(0.1), Random_Scale(), Gaussian_Blur(0.1), Normalize()])
    self.val_transforms   = transforms.Compose([ToTensor_Seg(), Normalize()])
    

  def setup(self, stage=None):
    self.train_dataset = Scan_Dataset_Segm(self.train_data_dir, transform = self.train_transforms)
    self.val_dataset   = Scan_Dataset_Segm(self.val_data_dir  , transform = self.val_transforms)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers = 3)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers = 3)