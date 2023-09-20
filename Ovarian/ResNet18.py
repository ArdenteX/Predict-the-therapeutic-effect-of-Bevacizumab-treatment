import torch
import torch.nn.functional as F
from MPNCOV.FASTMPNCOV import CovpoolLayer, SqrtmLayer, TriuvecLayer
from ECANET.ECA import eca_layer
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super(ResBlock, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(output_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(output_channel)
        )

        self.res_block = torch.nn.Sequential()
        if stride != 1 or input_channel != output_channel:
            self.res_block = torch.nn.Sequential(
                torch.nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(output_channel)
            )

    def forward(self, x):
        out = self.block(x)
        out += self.res_block(x)
        out = F.relu(out)
        return out


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.eca = eca_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channel = 64

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layers(ResBlock, 64, 2, stride=1)
        self.eca1 = eca_layer(3)
        self.layer2 = self.make_layers(ResBlock, 128, 2, stride=2)
        self.eca2 = eca_layer(3)
        self.layer3 = self.make_layers(ResBlock, 256, 2, stride=2)
        self.eca3 = eca_layer(3)
        self.eca4 = eca_layer(3)
        self.layer4 = self.make_layers(ResBlock, 512, 2, stride=2)
        self.layer_reduce = torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.layer_reduce_dn = torch.nn.BatchNorm2d(256)
        self.layer_reduce_relu = torch.nn.ReLU(inplace=True)
        # self.drop_out = torch.nn.Dropout2d(0.5)
        self.fc = torch.nn.Linear(int(256*(256+1)/2), num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def make_layers(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for s in strides:
            layers.append(block(self.in_channel, channels, stride=s))
            self.in_channel = channels

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.eca1(out)

        out = self.layer2(out)
        out = self.eca2(out)

        out = self.layer3(out)
        out = self.eca3(out)

        out = self.layer4(out)
        out = self.eca4(out)

        out = self.layer_reduce(out)
        out = self.layer_reduce_dn(out)
        out = self.layer_reduce_relu(out)

        out = CovpoolLayer(out)
        out = SqrtmLayer(out, 5)
        out = TriuvecLayer(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out


class MPNCOVResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MPNCOVResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer_reduce = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(int(256*(256+1)/2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 1x1 Conv. for dimension reduction
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)

        x = CovpoolLayer(x)
        x = SqrtmLayer(x, 5)
        x = TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def mpncovresnet18(n_classes=1_000, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = MPNCOVResNet(ECABasicBlock, [2, 2, 2, 2], num_classes=n_classes)
    return model


def mpncovresnet50(pretrained=False, n_classes=10000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param n_classes:
    """
    model = MPNCOVResNet(Bottleneck, [3, 4, 6, 3], n_classes)
    return model

