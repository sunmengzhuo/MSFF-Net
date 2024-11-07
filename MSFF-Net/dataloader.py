import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import config
import os
import pandas


class Dataset(data.Dataset):
    def __init__(self, data_path, label_path, mask_path, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.mask_path = mask_path
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        img_1 = Image.open(self.data_path[index]).convert('L')
        i = self.data_path[index][13:]
        i = i[:-4]
        img_2 = Image.open(self.mask_path + '/' + i + '.png').convert('L')
        img_1 = np.array(img_1).astype(float)
        img_2 = np.array(img_2).astype(float)
        img_2[img_2 < 244] = -0.5
        img_2[img_2 > 244] = 0.5
        img_1 = torch.from_numpy(img_1)
        img_2 = torch.from_numpy(img_2)
        label_data = pandas.read_excel(os.path.join(self.label_path))
        label = torch.tensor(label_data.loc[label_data['num'] == int(i), ['label']].values.item())
        resize = transforms.Resize((config.img_w, config.img_h))
        img_1 = torch.unsqueeze(img_1, dim=0)
        img_2 = torch.unsqueeze(img_2, dim=0)
        img_1 = resize(img_1)
        img_2 = resize(img_2)
        img = torch.cat((img_1, img_2), dim=0)
        if self.transform is not None:
            img = self.transform(img)

        return img, img_2, label
