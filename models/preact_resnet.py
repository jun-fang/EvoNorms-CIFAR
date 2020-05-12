'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm_act import NormAct2d

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_act_layer='BN-ReLU'):
        super(PreActBlock, self).__init__()
        self.norm_act_1 = NormAct2d(in_planes, norm_act_layer, nonlinearity=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm_act_2 = NormAct2d(planes, norm_act_layer, nonlinearity=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.norm_act_1(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.norm_act_2(out)
        out = self.conv2(out)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_act_layer='BN-ReLU'):
        super(PreActBottleneck, self).__init__()
        self.norm_act_1 = NormAct2d(in_planes, norm_act_layer, nonlinearity=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.norm_act_2 = NormAct2d(planes, norm_act_layer, nonlinearity=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm_act_3 = NormAct2d(planes, norm_act_layer, nonlinearity=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.norm_act_1(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.norm_act_2(out)
        out = self.conv2(out)
        out = self.norm_act_3(out)
        out = self.conv3(out)
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, norm_act_layer='BN-ReLU', num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.norm_act_layer = norm_act_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm_act_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(norm_act_layer='BN-ReLU'):
    return PreActResNet(PreActBlock, [2,2,2,2], norm_act_layer=norm_act_layer)

def PreActResNet34(norm_act_layer='BN-ReLU'):
    return PreActResNet(PreActBlock, [3,4,6,3], norm_act_layer=norm_act_layer)

def PreActResNet50(norm_act_layer='BN-ReLU'):
    return PreActResNet(PreActBottleneck, [3,4,6,3], norm_act_layer=norm_act_layer)

def PreActResNet101(norm_act_layer='BN-ReLU'):
    return PreActResNet(PreActBottleneck, [3,4,23,3], norm_act_layer=norm_act_layer)

def PreActResNet152(norm_act_layer='BN-ReLU'):
    return PreActResNet(PreActBottleneck, [3,8,36,3], norm_act_layer=norm_act_layer)


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
