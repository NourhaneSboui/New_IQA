from scipy.signal import convolve2d
import torch
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
from PIL import Image
import random
import csv


def default_loader(path):
    image = tifffile.imread(path)
    image = Image.fromarray(image)
    image = image.convert('L')
    return image

def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_array = np.array(patch)
    patch_mean = convolve2d(patch_array, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch_array), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch_array - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln

def NonOverlappingCropPatches(im, patch_size=32, stride=32):
    im_array = np.array(im)
    w, h = im_array.shape
    patches = []
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = im_array[i:i+patch_size, j:j+patch_size]
            patch = LocalNormalization(patch)
            patches.append(patch)
    return patches



def read_csv(csv_file):
            image_names = []
            mos_scores = []
            mos_stds = []

            with open(csv_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                   image_names.append(row['image_names'])
                   mos_scores.append(float(row['mos_scores']))
                   mos_stds.append(float(row['mos_stds']))

            return image_names, mos_scores, mos_stds


class IQADataset(Dataset):
    def __init__(self, conf, status='train'):
        self.loader = default_loader # Assign the loader function to self.loader
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        self.index=[]
        
        # Read image information from CSV file
        image_names, mos_scores, mos_stds =read_csv('C:/Users/win 10/Desktop/FR_IQA/IQA-optimization/Artif_MOS.csv')

        # Create indices based on image names
        num_images = len(image_names)
        indices = list(range(num_images))
        random.shuffle(indices)
        

        # Split the dataset into train, test, and validation sets
        train_ratio = 0.8
        test_ratio = 0.1

        trainindex = indices[:int(train_ratio * len(indices))]
        testindex = indices[int((1-test_ratio) * len(indices)):]
        train_index, val_index, test_index = [],[],[]
        for i in range(len(indices)):
            train_index.append(indices[i]) if (indices[i] in trainindex) else \
                test_index.append(indices[i]) if (indices[i] in testindex) else \
                    val_index.append(indices[i])
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('current status:',status)
            #print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('current status:',status)
            #print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))
            print('current status:',status)

        self.mos = [mos_scores[i] for i in self.index]
        self.mos_std = [mos_stds[i] for i in self.index]
        im_names = [image_names[i] for i in self.index]

        self.patches = ()
        self.label = []
        self.label_std = []
        for idx in range(len(self.index)):
            im = self.loader(os.path.join(im_dir, im_names[idx]))
            patches = NonOverlappingCropPatches(im, self.patch_size, self.stride)
            if status == 'train':
                self.patches = self.patches + tuple(patches) 
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
                    self.label_std.append(self.mos_std[idx])
            else:
                self.patches = tuple(self.patches) + (torch.stack(patches),) #
                self.label.append(self.mos[idx])
                self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], (torch.Tensor([self.label[idx]]), torch.Tensor([self.label_std[idx]]))
