import math

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import hub

from .utils import DropBlock2D, SplAtConv2d


class Bottleneck(M.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=1,
        cardinality=1,
        bottleneck_width=64,
        avd=False,
        avd_first=False,
        dilation=1,
        is_first=False,
        norm_layer=M.BatchNorm2d,
        dropblock_prob=0.0,
        last_gamma=False
    ):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = M.Conv2d(
            inplanes,
            group_width,
            kernel_size=1,
            bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = M.AvgPool2d(3, stride, padding=1, mode='average')
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)
        else:
            self.dropblock1 = M.Identity()
            self.dropblock2 = M.Identity()
            self.dropblock3 = M.Identity()

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        else:
            self.conv2 = M.Conv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False)
            self.bn2 = norm_layer(group_width)
            self.relu2 = M.ReLU()

        self.conv3 = M.Conv2d(
            group_width,
            planes * 4,
            kernel_size=1,
            bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            M.init.zeros_(self.bn3.weight)
        self.relu1 = M.ReLU()
        self.relu3 = M.ReLU()
        self.downsample = downsample if downsample is not None else M.Identity()
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropblock1(out)
        out = self.relu1(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.dropblock2(out)
            out = self.relu2(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropblock3(out)

        residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(M.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(
        self,
        block,
        layers,
        radix=1,
        groups=1,
        bottleneck_width=64,
        num_classes=1000,
        dilated=False,
        dilation=1,
        deep_stem=False,
        stem_width=64,
        avg_down=False,
        avd=False,
        avd_first=False,
        final_drop=0.0,
        dropblock_prob=0,
        last_gamma=False,
        norm_layer=M.BatchNorm2d
    ):
        super(ResNet, self).__init__()

        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        if deep_stem:
            self.conv1 = M.Sequential(
                M.Conv2d(3, stem_width, kernel_size=3, stride=2,
                         padding=1, bias=False),
                norm_layer(stem_width),
                M.ReLU(),
                M.Conv2d(stem_width, stem_width, kernel_size=3,
                         stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                M.ReLU(),
                M.Conv2d(stem_width, stem_width*2, kernel_size=3,
                         stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = M.Conv2d(3, 64, kernel_size=7,
                                  stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = M.ReLU()
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.avgpool = M.AdaptiveAvgPool2d(oshp=(1, 1))
        self.drop = M.Dropout(final_drop) if final_drop > 0.0 else M.Identity()
        self.fc = M.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                M.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(M.AvgPool2d(
                        kernel_size=stride, stride=stride))
                else:
                    down_layers.append(M.AvgPool2d(kernel_size=1, stride=1))
                down_layers.append(M.Conv2d(self.inplanes, planes * block.expansion,
                                            kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(M.Conv2d(self.inplanes, planes * block.expansion,
                                            kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = M.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise ValueError("unsupported dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return M.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = F.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)

        return x


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/37/files/2b402af4-5721-49d1-97a8-8216a13b2eae"
)
def resnest50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model


def resnest101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model


def resnest200(**kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model


def resnest269(**kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model


if __name__ == '__main__':
    import numpy as np
    model = resnest50()
    inp = mge.tensor(np.random.rand(2, 3, 224, 224))
    out = model(inp)
    print(out.shape)
