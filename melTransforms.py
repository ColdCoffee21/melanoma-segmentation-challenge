import numpy as np
from scipy.ndimage import rotate
import torch
from torchvision import transforms
from torchvision.transforms import functional as T_F

class Random_Rotate_Seg(object):
  """Rotate ndarrays in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      angle = float(torch.randint(low=-10, high=11, size=(1,)))
      image = rotate(image, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
      mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
    return {'image': image.copy(), 'mask': mask.copy()}

class ToTensor_Seg(object):
  """applies ToTensor for dict input"""
  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    image = transforms.ToTensor()(image)
    mask = transforms.ToTensor()(mask)
    return {'image': image.clone(), 'mask': mask.clone()}
  
# added class to horizontally flip images randomly
class Random_Horizontal_Flip_Seg(object):
  """Horizontal flip in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      image = np.fliplr(image)
      mask = np.fliplr(mask)
    return {'image': image.copy(), 'mask': mask.copy()}
  
# added class to vertically flip images randomly
class Random_Vertical_Flip_Seg(object):
  """Vertical flip in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      image = np.flipud(image)
      mask = np.flipud(mask)
    return {'image': image.copy(), 'mask': mask.copy()}
  
# added class to add color jitter to (only) images randomly
class Color_Jitter(object):
  """Color Jitter."""
  def __init__(self, probability, brightness_min=0.5, brightness_max='1.5', contrast=1, saturation_min=0.5, saturation_max=1.5, hue_min=-0.1, hue_max=0.1):
    self.probability = probability
    self.transform = transforms.ColorJitter(brightness=(brightness_min,brightness_max),contrast=(contrast),saturation=(saturation_min,saturation_max),hue=(hue_min,hue_max))
  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      image = self.transform(image)
    return {'image': image.clone(), 'mask': mask.clone()}
  
# added class to scale images randomly
class Random_Scale(object):
    def __init__(self, min_scale=1.0, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        scale = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(1)
        return {'image': T_F.affine(image.clone(), angle=0, translate=(0, 0), scale=scale, shear=0), 'mask': T_F.affine(mask.clone(), angle=0, translate=(0, 0), scale=scale, shear=0)}
    
# added class to blur images randomly
class Gaussian_Blur(object):
  def __init__(self, probability,  kernel_size=3, sigma=(0.1, 2.0)):
    self.probability = probability
    self.transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      image = self.transform(image)
    return {'image': image.clone(), 'mask': mask.clone()}
  
# added class to normalize images
class Normalize(object):
    def __init__(self, mean=[0.485], std=[0.229]):
        self.mean = mean
        self.std = std
        self.transform = transforms.Normalize(mean=mean, std=std)


    def __call__(self, sample):
      image, mask = sample['image'], sample['mask']
      image = self.transform(image)
      return {'image':image.clone(), 'mask':mask.clone()}