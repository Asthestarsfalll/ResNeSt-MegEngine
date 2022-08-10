import math
import megengine as mge
import megengine.functional as F
import megengine.module as M

class DropBlock2D(M.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Currently DropBlock2D is not implemented")


class SplAtConv2d(M.Module):

    def __init__(
        self, 
        in_channels, 
        channels, 
        kernel_size, 
        stride=(1, 1), 
        padding=(0, 0),
        dilation=(1, 1), 
        groups=1, 
        bias=True,
        radix=2, 
        reduction_factor=4,
        norm_layer=M.BatchNorm2d,
        dropblock_prob=0.0, 
    ):
        super(SplAtConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        self.conv = M.Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                            groups=groups*radix, bias=bias)
        use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels * radix) if use_bn else M.Identity()
        self.relu = M.ReLU()

        self.gap = M.AdaptiveAvgPool2d((1, 1))
        self.fc1 = M.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        
        self.bn1 = norm_layer(inter_channels) if use_bn else M.Identity()

        self.fc2 = M.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        
        self.dropblock = DropBlock2D(dropblock_prob, 3) if dropblock_prob > 0.0 else M.Identity()

        self.rsoftmax = rSoftMax(radix, groups) if radix > 1 else M.Sigmoid()
    
    def forward(self, x):
        return_dict = {}
        x = self.conv(x)
        x = self.bn0(x)
        x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = F.split(x, self.radix, axis=1)
            gap = sum(splited)
        else:
            gap = x

        gap = self.gap(gap)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).reshape(batch, -1, 1, 1)

        if self.radix > 1:
            attens = F.split(atten, self.radix, axis=1)
            return sum([att*split for (att, split) in zip(attens, splited)])
        return atten * x


class rSoftMax(M.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality


    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, self.cardinality, self.radix, -1).transpose((0, 2, 1, 3))
        x = F.softmax(x, axis=1)
        x = x.reshape(batch, -1)
        return x
