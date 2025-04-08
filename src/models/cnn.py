import torch
import torch.nn as nn
from torchvision import models
from .simple_dense_net import SimpleDenseNet

class ResNet(nn.Module):
    """adapted from https://pytorch.org/vision/main/models/resnet.html"""
    def __init__(self, variant, n_input_channels, num_classes=None, pool=True, pretrained=False):
        super(ResNet, self).__init__()
        if variant == '18' or variant == 18:
            self.net = models.resnet18(pretrained=pretrained)
        elif variant == '34' or variant == 34:
            self.net = models.resnet34(pretrained=pretrained)
        elif variant == '50' or variant == 50:
            self.net = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f'Invalid variant {variant} for ResNet.')
        # replace first and last layer
        self.net.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pool is None or not pool:
            # rm avgpool
            self.net.avgpool = nn.Identity()
        elif isinstance(pool, (nn.Module, nn.Sequential)):
            self.net.avgpool = pool
        if num_classes:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        else:
            self.net = nn.Sequential(*list(self.net.children())[:-1])

    def forward(self, x):
        return self.net(x)

class SimpleConvNet(nn.Module):
    def __init__(self, n_input_channels, out_channels:list[int], kernel_sizes, strides=1, paddings=0, dilations=1,
                 poolings=[], pool_strides=None, pool_dilation=1, pool_type='avg', bias=True, activation=nn.ReLU(), batch_norms=False, dropouts=0,
                 conv_type='conv2d', flatten=True, linear=None):
        """len of list determines number of layers, otherwise is repeated for all layers"""
        super(SimpleConvNet, self).__init__()
        nr_layers = len(out_channels)
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes]*nr_layers
        if not isinstance(strides, list):
            strides = [strides]*nr_layers
        if not isinstance(paddings, list):
            paddings = [paddings]*nr_layers
        if not isinstance(dilations, list):
            dilations = [dilations]*nr_layers
        if not isinstance(poolings, list):
            poolings = [poolings]*nr_layers
        if pool_strides is not None:
            if not isinstance(pool_strides, list):
                pool_strides = [pool_strides]*nr_layers
        if not isinstance(pool_dilation, list):
            pool_dilation = [pool_dilation]*nr_layers
        if not isinstance(batch_norms, list):
            batch_norms = [batch_norms]*nr_layers
        if not isinstance(dropouts, list):
            dropouts = [dropouts]*nr_layers

        if nr_layers != len(kernel_sizes) != len(strides) != len(paddings) != len(dilations) != len(poolings) != len(batch_norms) != len(dropouts):
            raise ValueError(f'Lists for layers do not work out for SimpleConvNet.')

        if pool_type not in ['avg', 'max']:
            raise ValueError(f'Invalid pool_type {pool_type} for SimpleConvNet.')
        self.conv_type = conv_type
        conv_type = conv_type.lower()
        if conv_type == 'conv1d':
            conv = nn.Conv1d
            pool = nn.MaxPool1d if pool_type == 'max' else nn.AvgPool1d
            batch_norm = nn.BatchNorm1d
        elif conv_type == 'conv2d':
            conv = nn.Conv2d
            pool = nn.MaxPool2d if pool_type == 'max' else nn.AvgPool2d
            batch_norm = nn.BatchNorm2d
        elif conv_type == 'conv3d':
            conv = nn.Conv3d
            pool = nn.MaxPool3d if pool_type == 'max' else nn.AvgPool3d
            batch_norm = nn.BatchNorm3d
        else:
            raise ValueError(f'Invalid conv_type {conv_type} for SimpleConvNet.')

        # construct blocks
        layers = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                layers.append(conv(n_input_channels, out_channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i], bias=bias))
            else:
                layers.append(conv(out_channels[i-1], out_channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i], bias=bias))
            if batch_norms[i]:
                layers.append(batch_norm(out_channels[i]))
            layers.append(activation)

            if dropouts[i]:
                layers.append(nn.Dropout(dropouts[i]))

            if poolings[i]:
                if pool_type == 'max':
                    layers.append(pool(kernel_size=poolings[i], stride=pool_strides[i] if pool_strides else poolings[i], dilation=pool_dilation[i]))
                else:
                    layers.append(pool(kernel_size=poolings[i], stride=pool_strides[i] if pool_strides else poolings[i]))

        # add linear layer
        if flatten:
            layers.append(nn.Flatten())
        if linear:
            if not isinstance(linear, SimpleDenseNet):
                raise ValueError(f'Invalid linear {linear} for SimpleConvNet.')
            layers.append(linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if '3d' in self.conv_type:
            # change c and t axis [b,t,c,h,w] -> [b,c,t,h,w]
            x = x.permute(0, 2, 1, 3, 4)
        return self.net(x)
