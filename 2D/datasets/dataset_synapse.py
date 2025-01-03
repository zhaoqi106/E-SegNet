import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch

def normalize_to(image):
    min_val = np.min(image)
    max_val = np.max(image)
    image_normalized = (image - min_val) / (max_val - min_val)
    return image_normalized

def visualize_image(image, title=None):
    if isinstance(image, Image.Image):  # If the image is a PIL image
        plt.imshow(image)
        plt.axis('off')

    elif isinstance(image, np.ndarray):  # If the image is a numpy array
        if image.ndim == 2:  # Grayscale image
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.axis('off')

    elif isinstance(image, torch.Tensor):  # If the image is a torch.Tensor
        if image.ndimension() == 3:
            # Assume the tensor is in CxHxW format (e.g., 3x224x224 for color image)
            image = image.permute(1, 2, 0)  # Convert to HxWxC for visualization
        image = image.cpu().numpy()  # Convert to numpy for plotting
        if image.ndim == 2:  # Grayscale image
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.axis('off')

    else:
        raise TypeError("Unsupported image type. Must be PIL, numpy, or torch.Tensor.")

    if title:
        plt.title(title)
    plt.show()

def random_crop(image, label):
    img_height, img_width = image.shape[:2]
    crop_height = int(img_height *0.2)
    crop_width = int(img_width *0.2)
    crop_height = img_height - crop_height
    crop_width = img_width - crop_width

    top = np.random.randint(0, img_height - crop_height + 1)
    left = np.random.randint(0, img_width - crop_width + 1)

    image = image[top:top + crop_height, left:left + crop_width]
    label = label[top:top + crop_height, left:left + crop_width]
    image = zoom(image, (img_height / crop_height, img_width / crop_width), order=3)  # why not 3?
    label = zoom(label, (img_height / crop_height, img_width / crop_width), order=0)
    return image,label


import numpy as np
from PIL import Image
import torchvision.transforms.functional as F


def random_affine_grayscale(image, label):
    if isinstance(image, np.ndarray):
        image = (normalize_to(image) * 255).astype(np.uint8)
        image = Image.fromarray(image)
    if isinstance(label, np.ndarray):
        label = label.astype(np.uint8)
        label = Image.fromarray(label,mode="L")

    angle = np.random.uniform(-30, 30)
    translate = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
    scale = np.random.uniform(0.8, 1.2)
    shear = np.random.uniform(-10, 10)

    image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)
    label = F.affine(label, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=Image.NEAREST,fill=0)

    brightness_factor = np.random.uniform(0.8, 1.2)
    contrast_factor = np.random.uniform(0.8, 1.2)

    if random.random() > 0.5:
        image = F.adjust_brightness(image, brightness_factor)
    if random.random() > 0.5:
        image = F.adjust_contrast(image, contrast_factor)

    image,label = np.array(image) / 255.0,np.array(label)

    return image,label


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if random.random() > 0.5:
            image,label = random_affine_grayscale(image,label)
        else:
            image = normalize_to(image)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class RandomGenerator2(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            image = normalize_to(image)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None,is_trans=True,transform2=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.is_trans = is_trans
        if transform2 is not None:
            self.transform2 = transform2

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform and self.split in ["train", "valid"]:
            if self.is_trans:
                sample = self.transform(sample)
            else:
                sample = self.transform2(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


