from omegaconf import DictConfig
import sys
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torchvision
# from models import resnet, densenet, mobilenet, googlenet
import models

class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)


class Resizer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.interpolate_mode = cfg.resizer.interpolate_mode
        self.scale_factor = cfg.data.image_size / cfg.data.resizer_image_size

        n = cfg.resizer.num_kernels
        r = cfg.resizer.num_resblocks
        slope = cfg.resizer.negative_slope

        self.module1 = nn.Sequential(
            nn.Conv2d(cfg.resizer.in_channels, n, kernel_size=7, padding=3),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(n, n, kernel_size=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.BatchNorm2d(n)
        )

        resblocks = []
        for i in range(r):
            resblocks.append(ResBlock(n, slope))
        self.resblocks = nn.Sequential(*resblocks)

        self.module3 = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n)
        )

        self.module4 = nn.Conv2d(n, cfg.resizer.out_channels, kernel_size=7,
                                 padding=3)

        self.interpolate = partial(F.interpolate,
                                   scale_factor=self.scale_factor,
                                   mode=self.interpolate_mode,
                                   align_corners=False,
                                   recompute_scale_factor=False)

    def forward(self, x):
        residual = self.interpolate(x)

        out = self.module1(x)
        out_residual = self.interpolate(out)

        out = self.resblocks(out_residual)
        out = self.module3(out)
        out = out + out_residual

        out = self.module4(out)

        out = out + residual

        return out

def get_base_model(dataset, arch='resnet50', in_channels=3, num_classes=10):
    assert arch in ['resnet50', 'densenet121', 'inceptionv3', 'mobilenetv2'], 'arch {} not supported'.format(arch)
    if dataset == 'cifar10':
        if arch == 'resnet50':
            base_model = models.resnet.ResNet50()
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        elif arch == 'densenet121':
            base_model = models.densenet.DenseNet121()
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        elif arch == 'inceptionv3':
            base_model = models.googlenet.GoogLeNet()
            base_model.pre_layers[0] = nn.Conv2d(in_channels, 192, kernel_size=3, padding=1)
        elif arch == 'mobilenetv2':
            base_model = models.mobilenet.MobileNetV2()
            base_model.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)

    elif dataset.startswith('image'):
        if arch == 'resnet50':
            base_model = torchvision.models.resnet50(num_classes=num_classes)
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
        elif arch == 'densenet121':
            base_model = torchvision.models.densenet121(num_classes=num_classes)
        elif arch == 'inceptionv3':
            # base_model = models_imagenet.googlenet.GoogLeNet()
            base_model = torchvision.models.GoogLeNet(num_classes=num_classes)
            # base_model.pre_layers[0] = nn.Conv2d(in_channels, 192, kernel_size=3, padding=1)
        elif arch == 'mobilenetv2':
            base_model = torchvision.models.mobilenet_v2(num_classes=num_classes)

    return base_model

def get_model(name, cfg):
    if name == "resizer":
        return Resizer(cfg)
    elif name == "base_model":
        if cfg.apply_resizer_model:
            in_channels = cfg.resizer.out_channels
        else:
            in_channels = cfg.resizer.in_channels
        return get_base_model(cfg.data.name, cfg.trainer.arch, in_channels, cfg.data.num_classes)
    else:
        raise ValueError(f"Incorrect name={name}. The valid options are"
                         "('resizer', 'base_model')")

class classification_model(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        if cfg.apply_resizer_model:
            self.resizer_model = get_model('resizer', cfg)
        else:
            self.resizer_model = None
        
        self.base_model = get_model('base_model', cfg)

        if cfg.apply_oss_resizer:
            self.oss_resizer = partial(F.interpolate,
                                   scale_factor=cfg.data.image_size / cfg.data.resizer_image_size,
                                   mode=cfg.oss_resizer.interpolate_mode,
                                   align_corners=False,
                                   recompute_scale_factor=False)
        else:
            self.oss_resizer = None
    
    def forward(self, x):
        if self.resizer_model is not None:
            x = self.resizer_model(x)
        elif self.oss_resizer is not None:
            x = self.oss_resizer(x)
        x = self.base_model(x)
        return x