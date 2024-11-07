import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch
from torch.utils import data
import pandas
import config

class Dataset(data.Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        # 返回图片数
        return len(self.data_path)

    def __getitem__(self, index):
        img = Image.open(self.data_path[index]).convert('L')
        i = self.data_path[index][13:]
        i = i[:-4]
        img = np.array(img).astype(float)
        # img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0)
        label_data = pandas.read_excel(os.path.join(self.label_path))
        label = torch.tensor(label_data.loc[label_data['住院号'] == int(i), ['结局']].values.item())
        if self.transform is not None:
            img = self.transform(img)

        return img, label


data_path = './data'
label_path = './data/腹膜转移_finals.xlsx'

# 图像预处理
other_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def data_list(datapath: str):
    train_images_path = []
    val_images_path = []
    test_images_path = []
    img_path = [datapath + '/train', datapath + '/test_']
    for path in img_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                if path == datapath + '/train':
                    train_images_path.append(os.path.join(root, file))
                # elif path == datapath+'/valid':
                #     val_images_path.append(os.path.join(root, file))
                else:
                    test_images_path.append(os.path.join(root, file))
    return train_images_path, test_images_path


train_images_path, test_images_path = data_list(data_path)

# 计算训练集的均值、方差（训练集的数量为331）
train_data = Dataset(train_images_path, label_path, transform=other_transform)
train_loader = DataLoader(dataset=train_data, batch_size=331, shuffle=True)
train = next(iter(train_loader))[0]
train_mean = np.mean(train.numpy(), axis=(0, 2, 3))
train_std = np.std(train.numpy(), axis=(0, 2, 3))


print("train_mean:", train_mean)
print("train_std:", train_std)
