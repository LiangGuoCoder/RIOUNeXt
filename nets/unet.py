import torch
import torch.nn as nn
from nets.EncoderNet import Net
from nets.fusion import AFF, iAFF, DAF


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, fuse_type='AFF'):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=2, dilation=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.relu   = nn.ReLU(inplace = True)
        self.PRelu = nn.PReLU(num_parameters=1)
        if fuse_type == 'AFF':
            self.fuse_mode = AFF(channels=out_size)
        elif fuse_type == 'iAFF':
            self.fuse_mode = iAFF(channels=out_size)
        elif fuse_type == 'DAF':
            self.fuse_mode = DAF()
        else:
            self.fuse_mode = None

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # identity = self.conv3(outputs)

        outputs = self.conv1(outputs)
        outputs = self.PRelu(outputs)

        outputs = self.conv2(outputs)
        # outputs += identity
        # outputs = self.fuse_mode(outputs, identity)
        outputs = self.PRelu(outputs)

        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, backbone='resnet50', fuse_type='iAFF', fuse_type_1='AFF'):
        super(Unet, self).__init__()

        if backbone == "resnet50":
            self.resnet = Net(fuse_type=fuse_type, fuse_type_1=fuse_type_1)
            # self.resnet = resnet50(fuse_type=fuse_type)
            # self.resnet = resnet50(pretrained = pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        global feat1, feat2, feat3, feat4, feat5
        if self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
