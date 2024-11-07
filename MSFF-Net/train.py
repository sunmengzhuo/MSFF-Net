import torch
import os
import argparse

from dataloader import Dataset
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from models import model
import pandas as pd


def data_list(datapath: str):
    train_images_path = []
    test_images_path = []
    mask_images_path = []
    img_path = [datapath + '/train', datapath + '/test_']
    for path in img_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                if path == datapath + '/train':
                    train_images_path.append(os.path.join(root, file))
                else:
                    test_images_path.append(os.path.join(root, file))
    return train_images_path, test_images_path


def train_on_epochs(train_dataset: Dataset, test_dataset: Dataset, pre_train: str):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    net = model.resnet18()

    if pre_train != '':
        net.load_state_dict(torch.load(pre_train), strict=False)

    net.to(device)

    model_params = net.parameters()

    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma,
                                                last_epoch=-1)

    loss_function = nn.CrossEntropyLoss()

    writer = SummaryWriter('./Result')

    train_num = len(train_dataset)
    train_root = './data/train'
    train = []
    label_list = []
    for root, dirs, files in os.walk(train_root):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                train.append(os.path.join(root, file))
    for img_path in train:
        i = img_path[13:]
        i = i[:-4]
        label_root = './data/pm_finals.xlsx'
        label_data = pd.read_excel(os.path.join(label_root))
        label = label_data.loc[label_data['num'] == int(i), ['label']].values.item()
        label_list.append(label)
    # train_weights = [1/281 if label == 0 else 1/50 for label in label_list]

    # sampler = WeightedRandomSampler(train_weights, 331, replacement=True)
    train_loader = DataLoader(train_dataset, **config.dataset_params)

    train_steps = len(train_loader)

    for epoch in range(config.epoches):
        # train
        net.train()
        running_loss = 0.0
        test_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, (imgs, masks, labs) in enumerate(train_bar):
            images = imgs.float()
            labels = labs.to(torch.int64)
            mask = masks.float()
            optimizer.zero_grad()
            logits = net(images.to(device), mask.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     config.epoches,
                                                                     loss)
        scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        net.eval()
        with torch.no_grad():
            print('[epoch %d] train_loss: %.3f' %
                  (epoch + 1, running_loss / train_steps))
            writer.add_scalar("TrainLoss", running_loss / train_steps, epoch)
            torch.save(net.state_dict(), config.save_path + 'ResNet18-' + str(epoch) + '.pth')


def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='./data')
    parser.add_argument('-l', '--label_path', help='path to your datasets label', default='./data/pm_finals.xlsx')
    parser.add_argument('-m', '--mask_path', help='path to your mask', default='./mask')
    parser.add_argument('-r', '--pre_train', help='path to the pretrain weights', default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    label_path = args.label_path
    mask_path = args.mask_path
    pre_train = args.pre_train
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.Normalize([0.14848179], [0.25421804])
        # 肿瘤未直方图均衡化
    ])
    train_images_path, test_images_path = data_list(data_path)
    train_on_epochs(Dataset(train_images_path, label_path, mask_path, transform=None),
                    Dataset(test_images_path, label_path, mask_path, transform=None),
                    pre_train)
