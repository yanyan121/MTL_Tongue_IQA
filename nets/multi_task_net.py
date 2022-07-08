##
import torch
import torch.nn as nn
from resnet import resnet18
from vgg16 import VGG
import torch
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, p):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)
        self.drop   = nn.Dropout(p=p)
        # nn.Dropout(p=0.5)对应到每一个像素 nn.Dropout2d(p=p)对应某一维度

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.drop(outputs)
        return outputs

class Cla_model_single(nn.Module):
    def __init__(self):
        super(Cla_model_single, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.load_state_dict(resnet18)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.model.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, 2)

    def forward(self, inputs):
        # norm_mat = torch.mean(inputs.reshape((inputs.size(0), -1)), dim=1).reshape((inputs.size(0), 1, 1, 1))
        # inputs_norm = inputs / norm_mat
        # x_c = self.model.conv1(inputs_norm)
        x_c = self.model.conv1(inputs)
        x_c = self.model.bn1(x_c)
        x_c = self.model.relu(x_c)
        x_c = self.model.maxpool(x_c)
        x_c = self.model.layer1(x_c)
        x_c = self.model.layer2(x_c)
        x_c = self.model.layer3(x_c)
        x_c_convlayer4 = self.model.layer4(x_c)

        x_c = self.model.avgpool(x_c_convlayer4)
        x_c = x_c.view(x_c.size(0), -1)
        out_c = self.model.fc(x_c)
        results = {'mean': out_c, 'convlayer4': x_c_convlayer4}
        #             return results
        return results

class multi_model_unet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='resnet18', uncertainty='normal', n_samples=5, p=0.0):
        super(multi_model_unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet18":
            self.resnet = resnet18(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet18.'.format(backbone))
        out_filters = [64, 128, 256, 512]
        # upsampling_1
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3], p=p)
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2], p=p)
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1], p=p)
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0], p=p)
        # upsampling_2
        # 64,64,512
        self.up_concat4_2 = unetUp(in_filters[3], out_filters[3], p=p)
        # 128,128,256
        self.up_concat3_2 = unetUp(in_filters[2], out_filters[2], p=p)
        # 256,256,128
        self.up_concat2_2 = unetUp(in_filters[1], out_filters[1], p=p)
        # 512,512,64
        self.up_concat1_2 = unetUp(in_filters[0], out_filters[0], p=p)
        if backbone == 'resnet18':
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

        self.uncertainty = uncertainty
        self.n_samples = n_samples
        self.backbone = backbone

        ## classification
        self.cla_net = Cla_model_single()

        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.model.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, 2)

        # self.sa_seg = SelfAttention(1024, 'relu')
        # self.sa_cla = SelfAttention(1024, 'relu')
        # self.concat = torch.cat()
        self.cat_conv_seg = nn.Conv2d(1024, 512, 3, 1, 1)
        self.cat_conv_cla = nn.Conv2d(1024, 512, 3, 1, 1)

    def forward(self, inputs):  # 嵌套整个网络模型函数，方便调用
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet18":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        results_cla = self.cla_net.forward(inputs)
        convlayer4_cla = results_cla['convlayer4']
        share_cat = torch.cat((feat5, convlayer4_cla), 0)
        ## cla
        cat_cla_out1 = self.sa_cla(share_cat)
        cat_cla_out2 = self.cat_conv_cla(cat_cla_out1)
        ## image-level  classification
        x_cla = self.model.avgpool(cat_cla_out2)
        x_cla = x_cla.view(x_cla.size(0), -1)
        mean_cla = self.model.fc(x_cla)

        ## seg ######
        cat_seg_out1 = self.sa_seg(share_cat)
        cat_seg_out2 = self.cat_conv_cla(cat_seg_out1)
        ## 后续紧跟 分类（语义分类）

        # mean 分割decoder
        up4 = self.up_concat4(feat4, cat_seg_out2)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        # segmentation
        if self.up_conv != None:
            up1 = self.up_conv(up1)
        mean_seg = self.final(up1)

        results = {'mean_cla': mean_cla, 'mean_seg': mean_seg}
        return results