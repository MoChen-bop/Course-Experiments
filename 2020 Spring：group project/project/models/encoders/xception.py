import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.model_zoo as model_zoo
from torch.nn import init


__all__ = ['Xception41', 'Xception64', 'Xception71']


def check_data(data, number):
    if type(data) == int:
        return [data] * number
    assert len(data) == number
    return data


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, 
        dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
            padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_fist=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_fist:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_fist:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)


    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    
    def __init__(self, bottleneck_params, num_classes=1000):
        super(Xception, self).__init__()

        self.conv1 = nn.Conv2d( 3, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        in_channel = 64
        self.entry_flow, in_channel = self.block_flow(
            block_num = bottleneck_params['entry_flow'][0],
            strides = bottleneck_params['entry_flow'][1],
            chns = bottleneck_params['entry_flow'][2], 
            in_channel=in_channel)

        self.middle_flow, in_channel = self.block_flow(
            block_num = bottleneck_params['middle_flow'][0],
            strides = bottleneck_params['middle_flow'][1],
            chns = bottleneck_params['middle_flow'][2], 
            in_channel=in_channel)

        self.exit_flow, in_channel = self.exit_block_flow(
            block_num = bottleneck_params['exit_flow'][0],
            strides = bottleneck_params['exit_flow'][1],
            chns = bottleneck_params['exit_flow'][2], 
            in_channel=in_channel)

        self.fc_1 = nn.Linear(in_channel, 1024)
        self.fc_2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: batch_size x 3 x 512 x 512
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x) # batch_size x 64 x 256 x 256

        x = self.entry_flow(x) # batch_size x 728 x 32 x 32

        x = self.middle_flow(x) # batch_size x 728 x 32 x 32

        x = self.exit_flow(x) # batch_size x 2048 x 16 x 16

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) # batch_size x 2048

        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


    def block_flow(self, block_num, strides, chns, in_channel):
        block = []
        strides = check_data(strides, block_num)
        chns = check_data(chns, block_num)
        for i in range(block_num):
            out_channel = chns[i]
            block.append(Block(in_filters=in_channel, out_filters=out_channel, reps=3,
                strides=strides[i], start_with_relu=True, grow_fist=True))
            in_channel = out_channel

        return nn.Sequential(*block), in_channel


    def exit_block_flow(self, block_num, strides, chns, in_channel):
        assert block_num == 2

        block = []
        block.append(Block(in_channel, chns[0][0], 1, strides[0][0], True, True))
        block.append(Block(chns[0][0], chns[0][1], 1, strides[0][1], True, True))
        block.append(Block(chns[0][1], chns[0][2], 1, strides[0][2], True, True))

        block.append(Block(chns[0][2], chns[1][0], 1, strides[1][0], True, True))
        block.append(Block(chns[1][0], chns[1][1], 1, strides[1][1], True, True))
        block.append(Block(chns[1][1], chns[1][2], 1, strides[1][2], True, True))

        return nn.Sequential(*block), chns[1][2]
    

    def name(self):
        return 'Xception'


def Xception41(num_classes):
    bottleneck_params = {
        'entry_flow':  (3, [2, 2, 2], [128, 256, 728]),
        'middle_flow': (8, 1, 728),
        'exit_flow': (2, [[2, 1, 1], [1, 1, 1]], [[728, 1024, 1024], [1536, 1536, 2048]])
    }
    return Xception(bottleneck_params, num_classes)


def Xception64(num_classes):
    bottleneck_params = {
        'entry_flow':  (3, [2, 2, 2], [128, 256, 728]),
        'middle_flow': (16, 1, 728),
        'exit_flow': (2, [[2, 1, 1], [1, 1, 1]], [[728, 1024, 1024], [1536, 1536, 2048]])
    }
    return Xception(bottleneck_params, num_classes)


def Xception71(num_classes):
    bottleneck_params = {
        'entry_flow':  (5, [2, 1, 2, 1, 2], [128, 256, 256, 728, 728]),
        'middle_flow': (16, 1, 728),
        'exit_flow': (2, [[2, 1, 1], [1, 1, 1]], [[728, 1024, 1024], [1536, 1536, 2048]])
    }
    return Xception(bottleneck_params, num_classes)


if __name__ == '__main__':
    
    batch_image = torch.randn((4, 3, 512, 512))
    xception = Xception41(1000)
    predict = xception(batch_image)
    print(predict.shape)