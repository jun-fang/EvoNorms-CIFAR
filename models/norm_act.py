import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

# references: 
# https://arxiv.org/pdf/2004.02967.pdf
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/normalization.py

class NormAct2d(nn.Module):

    def __init__(self, num_channels, norm_act_layer, nonlinearity=True):
        super(NormAct2d, self).__init__()
        norm_act_layer = norm_act_layer.lower()
        if norm_act_layer.startswith(('batch', 'bn')):
            self.norm_act = BatchNormReLU(num_channels, nonlinearity=nonlinearity)
        elif norm_act_layer.startswith(('group', 'gn')):
            self.norm_act = GroupNormReLU(num_channels, nonlinearity=nonlinearity)
        elif norm_act_layer.startswith(('evo', 'en')):
            if 's0' in norm_act_layer:
                self.norm_act = EvoNormS0(num_channels, nonlinearity=nonlinearity)
            elif 'b0' in norm_act_layer:
                self.norm_act = EvoNormB0(num_channels, nonlinearity=nonlinearity)
        else:
            raise ValueError('Expected BatchNorm, GroupNorm, EvoNormS0 or EvoNormB0!')

    def forward(self, x):
        return self.norm_act(x)


class BatchNormReLU(nn.Module):
    def __init__(self, num_channels, nonlinearity=True):
        super(BatchNormReLU, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_channels)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        if self.nonlinearity:
            return F.relu(self.batch_norm(x))
        else:
            return self.batch_norm(x)
            

class GroupNormReLU(nn.Module):
    def __init__(self, num_channels, nonlinearity=True, num_groups=32):
        super(GroupNormReLU, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        if self.nonlinearity:
            return F.relu(self.group_norm(x))
        else:
            return self.group_norm(x)


## EvoNorms
class EvoNormS0(nn.Module):
    def __init__(self, num_channels, nonlinearity=True, num_groups=32, eps=1e-5):
        super(EvoNormS0, self).__init__()
        self.num_channels = num_channels
        self.nonlinearity = nonlinearity
        if self.nonlinearity:
            self.v = Parameter(torch.Tensor(1, num_channels, 1, 1))
            self.num_groups = num_groups
            self.eps = eps
        self.gamma = Parameter(torch.Tensor(1, num_channels, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_channels, 1, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        if self.nonlinearity:
            nn.init.ones_(self.v)
        
    def forward(self, x):
        if self.nonlinearity:
            num = x * torch.sigmoid(self.v * x)
            return num / group_std(x, groups=self.num_groups, eps=self.eps) * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta


class EvoNormB0(nn.Module):

    def __init__(self, num_channels, nonlinearity=True, eps=1e-05, momentum=0.1, training=True):
        super(EvoNormB0, self).__init__()
        self.num_channels = num_channels
        self.nonlinearity = nonlinearity
        if self.nonlinearity:
            self.v = Parameter(torch.Tensor(1, num_channels, 1, 1))

        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(torch.Tensor(1, num_channels, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_channels, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_channels, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        if self.nonlinearity:
            nn.init.ones_(self.v)
        
    def forward(self, x):
        if self.training:
            var = torch.var(x, dim=(0, 2, 3), keepdim = True)
            self.running_var.mul_(1 - self.momentum)
            self.running_var.add_(self.momentum * var)
        else:
            var = self.running_var

        if self.nonlinearity:
            den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
            return x / den * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta


## Helper functions for EvoNorms
def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    return torch.sqrt(var + eps)


def group_std(x, groups=32, eps=1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True)
    std = torch.sqrt(var + eps)
    std = std.expand_as(x)
    return torch.reshape(std, (N, C, H, W))