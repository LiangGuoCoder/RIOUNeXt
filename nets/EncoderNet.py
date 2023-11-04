from torch import nn
from nets.fusion import AFF, iAFF, DAF
from modules.odconv import ODConv2d


def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EcaLayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3, groups=1,
                 base_width=64, norm_layer=None, fuse_type='iAFF', fuse_type_1='AFF',
                 reduction=0.0625, kernel_num=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        width = planes
        self.conv1 = conv1x1(width, planes * self.expansion)
        self.bn1 = norm_layer(planes * self.expansion)

        self.conv2 = odconv3x3(inplanes, width, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(planes * self.expansion, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.PRelu = nn.PReLU(num_parameters=1)
        self.GeLU = nn.GELU()

        self.downsample = downsample
        self.stride = stride

        if fuse_type == 'AFF':
            self.fuse_mode = AFF(channels=planes * self.expansion)
        elif fuse_type == 'iAFF':
            self.fuse_mode = iAFF(channels=planes * self.expansion)
        elif fuse_type == 'DAF':
            self.fuse_mode = DAF()
        else:
            self.fuse_mode = None

        if fuse_type_1 == 'AFF':
            self.fuse_mode_1 = AFF(channels=planes)
        elif fuse_type_1 == 'iAFF':
            self.fuse_mode_1 = iAFF(channels=planes)
        elif fuse_type_1 == 'DAF':
            self.fuse_mode_1 = DAF()
        else:
            self.fuse_mode_1 = None

    def forward(self, x):
        identity = x

        out = self.conv2(x)
        out = self.bn1(out)

        out = self.conv1(out)

        out = self.GeLU(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.fuse_mode(out, identity)  # iAFF

        out = self.GeLU(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, k_size=None, fuse_type='iAFF', fuse_type_1='AFF', reduction=0.0625, kernel_num=1):

        if k_size is None:
            k_size = [3, 3, 3, 3, 3]
        self.inplanes = 16
        super(ResNet, self).__init__()
        assert fuse_type in ['AFF', 'iAFF', 'DAF']
        assert fuse_type_1 in ['AFF', 'iAFF', 'DAF']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.PRelu = nn.PReLU(num_parameters=1)
        self.pool = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 16, layers[0], int(k_size[0]), fuse_type=fuse_type,
                                       fuse_type_1=fuse_type_1, reduction=reduction, kernel_num=kernel_num)

        self.layer1 = self._make_layer(block, 64, layers[1], int(k_size[1]), fuse_type=fuse_type,
                                       fuse_type_1=fuse_type_1, reduction=reduction, kernel_num=kernel_num)

        self.layer2 = self._make_layer(block, 128, layers[2], int(k_size[2]), stride=2, fuse_type=fuse_type,
                                       fuse_type_1=fuse_type_1, reduction=reduction, kernel_num=kernel_num)

        self.layer3 = self._make_layer(block, 256, layers[3], int(k_size[3]), stride=2, fuse_type=fuse_type,
                                       fuse_type_1=fuse_type_1, reduction=reduction, kernel_num=kernel_num)

        self.layer4 = self._make_layer(block, 512, layers[4], int(k_size[4]), stride=2, fuse_type=fuse_type,
                                       fuse_type_1=fuse_type_1, reduction=reduction, kernel_num=kernel_num)

    def _make_layer(self, block, planes, blocks, k_size, fuse_type, fuse_type_1, stride=1,
                    reduction=0.625, kernel_num=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size, fuse_type=fuse_type,
                            fuse_type_1=fuse_type_1, reduction=reduction, kernel_num=kernel_num))
        self.inplanes = planes * block.expansion
        self.fuse_mode_1 = iAFF(channels=512)
        self.fuse_mode_2 = iAFF(channels=1024)
        self.fuse_mode_3 = iAFF(channels=2048)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size, fuse_type=fuse_type, fuse_type_1=fuse_type_1,
                                reduction=reduction, kernel_num=kernel_num))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.PRelu(x)

        x = self.pool(feat1)
        feat2 = self.layer1(x)

        feat_3 = self.layer2(feat2)
        feat3 = self.layer2(feat2)
        feat3 = self.fuse_mode_1(feat3, feat_3)  # iAFF

        feat_4 = self.layer3(feat_3)
        feat4 = self.layer3(feat3)
        feat4 = self.fuse_mode_2(feat4, feat_4)  # iAFF

        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]

def Net(fuse_type='iAFF', fuse_type_1='AFF', k_size=None, **kwargs):
    if k_size is None:
        k_size = [3, 3, 3, 3, 3]
    model = ResNet(Bottleneck, [1, 1, 1, 3, 1], fuse_type=fuse_type, fuse_type_1=fuse_type_1, k_size=k_size, **kwargs)
    return model
