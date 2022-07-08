# import torch
# import torch.nn as nn
# from vgg import VGG16
#
#
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, inputs1, inputs2):
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1) # 主干层提取网络与加强层提取网络进行特征堆叠
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         return outputs
#
# class Unet(nn.Module):
#     def __init__(self, num_classes=21, in_channels=3, pretrained=False):
#         super(Unet, self).__init__()
#         self.vgg = VGG16(pretrained=pretrained,in_channels=in_channels)
#         in_filters = [192, 384, 768, 1024]
#         out_filters = [64, 128, 256, 512]
#         # upsampling
#         # 64,64,512
#         self.up_concat4 = unetUp(in_filters[3], out_filters[3])
#         # 128,128,256
#         self.up_concat3 = unetUp(in_filters[2], out_filters[2])
#         # 256,256,128
#         self.up_concat2 = unetUp(in_filters[1], out_filters[1])
#         # 512,512,64
#         self.up_concat1 = unetUp(in_filters[0], out_filters[0])
#
#         # final conv (without any concat)
#         self.final = nn.Conv2d(out_filters[0], num_classes, 1) ##构建1*1的卷积
#
#     def forward(self, inputs):
#         feat1 = self.vgg.features[  :4 ](inputs)
#         feat2 = self.vgg.features[4 :9 ](feat1)
#         feat3 = self.vgg.features[9 :16](feat2)
#         feat4 = self.vgg.features[16:23](feat3)
#         feat5 = self.vgg.features[23:-1](feat4)
#
#         up4 = self.up_concat4(feat4, feat5)
#         up3 = self.up_concat3(feat3, up4)
#         up2 = self.up_concat2(feat2, up3)
#         up1 = self.up_concat1(feat1, up2)
#
#         final = self.final(up1)
#
#         return final
#
#     def _initialize_weights(self, *stages):
#         for modules in stages:
#             for module in modules.modules():
#                 if isinstance(module, nn.Conv2d):
#                     nn.init.kaiming_normal_(module.weight)
#                     if module.bias is not None:
#                         module.bias.data.zero_()
#                 elif isinstance(module, nn.BatchNorm2d):
#                     module.weight.data.fill_(1)
#                     module.bias.data.zero_()
#


import torch
import torch.nn as nn

# from nets.resnet import resnet50
# import resnet50
from nets.resnet import resnet50,resnet18
from vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3,pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained,in_channels=in_channels)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet18":
            self.resnet = resnet18(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
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



        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        ## 通过1*1的卷积进行降维，对应输出的特征层进行分类

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet18":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        ## seg  ---------------------------##
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)


        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        ## 对应输出的特征层进行分类

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
