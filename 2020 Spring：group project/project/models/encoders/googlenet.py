import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['googlenet']


def googlenet(transform_input=False):
    model_url = 'https://download.pytorch.org/models/googlenet-1378be20.pth'
    model = GoogLeNetFeatureExtractor(transform_input=transform_input)
    state_dict = load_state_dict_from_url(model_url)
    model.load_state_dict(state_dict, strict=False)
    return model


def GoogLeNetFeatureExtractor(nn.Module):
    
    def __init__(self, transform_input=False):
        super().__init__()

        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=2)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x) # batch_size x 3 x 224 x 224

        x = self.conv1(x)    # batch_size x 64 x 112 x 112
        x = self.maxpool1(x) # batch_size x 64 x 56 x 56
        x = self.conv2(x)    # batch_size x 64 x 56 x 56

        x = self.conv3(x)    # batch_size x 192 x 56 x 56
        x = self.maxpool2(x) # batch_size x 192 x 28 x 28

        x = self.inception3a(x) # batch_size x 256 x 28 x 28
        x = self.inception3b(x) # batch_size x 480 x 28 x 28
        x = self.maxpool3(x)    # batch_size x 480 x 14 x 14

        x = self.inception4a(x) # batch_size x 512 x 14 x 14
        x = self.inception4b(x) # batch_size x 512 x 14 x 14
        x = self.inception4c(x) # batch_size x 512 x 14 x 14
        x = self.inception4d(x) # batch_size x 528 x 14 x 14
        x = self.inception4e(x) # batch_size x 832 x 14 x 14
        x = self.maxpool4(x)    # batch_size x 832 x  7 x  7

        x = self.inception5a(x) # batch_size x  832 x 7 x 7
        x = self.inception5b(x) # batch_size x 1024 x 7 x 7
        x = self.avgpool(x)     # batch_size x 1024 x 1 x 1

        x = torch.flatten(x, 1) # batch_size x 1024
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):

        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x),
                   self.branch3(x), self.branch4(x)]

        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)
