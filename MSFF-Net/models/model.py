import torch.nn as nn
import torch
from torchvision import transforms


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv11 = nn.Conv2d(2, self.in_channel, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # self.extra_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        # 上采样的三个方法，看哪个效果好选哪个 #
        # self.pyramid_conv1_3x3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.pyramid_conv1_2x2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.pyramid_conv1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 双线性插值
        self.pyramid_conv1_up_conv = nn.Conv2d(512, 256, 1, 1)
        # ----------------------------- #

        self.pyramid_conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.pyramid_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)


        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc1 = nn.Linear(512 * block.expansion, 256)
            self.fc2 = nn.Linear(256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def _resize(self, x, lenth, h, w):
        x = x.repeat(1, lenth, 1, 1)
        resize = transforms.Resize([h, w])
        x = resize(x)
        return x

    def forward(self, x, y):
        x = self.conv11(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = x + self._resize(y, x.shape[1], x.shape[2], x.shape[3])
        c1 = x
        x = self.layer2(x)
        x = x + self._resize(y, x.shape[1], x.shape[2], x.shape[3])
        c2 = x

        x = self.layer3(x)
        c3 = x

        x = self.layer4(x)

        # 获取不同层次特征
        c4 = x

        # 构建特征金字塔
        # 第一个方法
        # p4 = self.pyramid_conv1_3x3(c4)

        # 第二个方法
        # p4 = self.pyramid_conv1_2x2(c4)

        # 第三个方法
        p4 = self.pyramid_conv1_up(c4)
        p4 = self.pyramid_conv1_up_conv(p4)

        # 第四个方法
        # p4 = self.pyramid_conv1_up_conv(c4)
        # p4 = self.pyramid_conv1_up(p4)

        p3 = self.pyramid_conv2(c3)
        p2 = self.pyramid_conv3(c2)

        fused_feature = torch.cat((p4, p3, p2), dim=1)
        # 备选方案
        # p3 = p3 + nn.functional.interpolate(p4, scale_factor=2, mode='nearest')
        # p2 = p2 + nn.functional.interpolate(p3, scale_factor=2, mode='nearest')
        #
        # # 降采样
        # p2 = nn.functional.interpolate(p2, scale_factor=0.5, mode='nearest')
        #
        # # 使用额外的卷积层
        # p1 = self.extra_conv(c1)
        # p1 = nn.functional.interpolate(p1, scale_factor=0.25, mode='nearest')
        # # 融合所有尺度的特征
        # fused_feature = p1 + p2 + p3

        if self.include_top:
            x = self.avgpool(fused_feature)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc2(x)

        return x


def resnet18(num_classes=2, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=2, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=2, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=2, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=2, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=2, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
