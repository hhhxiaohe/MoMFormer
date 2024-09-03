# TODO:Creat dataset
import json
import logging
import os
from glob import glob
import cv2
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, n_input, j_dir, scale, query):
        self.n = n_input
        self.dir = j_dir
        self.scale = scale
        self.q = query
        assert os.path.exists(self.dir), "cannot find {} file".format(self.dir)
        self.data_dict = json.load(open(self.dir, 'r'))
        label_dir = self.data_dict[f'{self.q}_label']
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(label_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, image, scale):
        h, w = image.shape[:2]
        newH, newW = int(scale * h), int(scale * w)
        image = cv2.resize(image, (newH, newW))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        return image.transpose((2, 0, 1))

    def __getitem__(self, i):
        data_dict = self.data_dict
        idx = self.ids[i]
        output_dict = dict()

        # train set
        for j in range(1, 1+self.n):
            imgs_dir = data_dict[f'{j}_{self.q}_img']
            img_file = glob(imgs_dir + idx + "*")
            img_name = img_file[0].split('.')[-1]
            if img_name == 'tif':
                img = tiff.imread(img_file[0])
            else:
                img = cv2.imdecode(np.fromfile(img_file[0], dtype=np.uint8), -1)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.preprocess(img, self.scale)
            output_dict[f'image{j}'] = torch.from_numpy(img/255)  # normalized

        # val set
        mask_file = glob(data_dict[f'{self.q}_label'] + idx + "*")
        mask = cv2.imdecode(np.fromfile(mask_file[0], dtype=np.uint8), -1)
        mask = self.preprocess(mask, self.scale)
        output_dict['mask'] = torch.from_numpy(mask)

        return output_dict
