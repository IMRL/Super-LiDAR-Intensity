import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet, ResNet34_Weights
import torchvision.ops as ops


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    seq = nn.Sequential(*layers)
    for m in seq.modules():
        init_weights(m)
    return seq


def convt_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    seq = nn.Sequential(*layers)
    for m in seq.modules():
        init_weights(m)
    return seq


class ASPP_w_deform(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        self.atrous_blocks = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=True))
        self.conv_offset = nn.Conv2d(in_channels, 2*5*5, kernel_size=5, padding=2)
        self.deform_conv = ops.DeformConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.conv_1x1_out = nn.Conv2d(out_channels*(len(atrous_rates)+2), out_channels, kernel_size=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.batch_norm(self.conv_1x1(x)))
        out = [x1]
        for atrous_conv in self.atrous_blocks:
            out.append(self.relu(self.batch_norm(atrous_conv(x))))
        offset = self.conv_offset(x)
        out.append(self.relu(self.deform_conv(x, offset)))
        fea = torch.cat(out, dim=1)
        return self.conv_1x1_out(fea)


class DisCompensation(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_alpha = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.to_beta = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.to_gamma = nn.Conv2d(2, 1, kernel_size=5, padding=2)

    def forward(self, x):
        reflectance = x[:, 0:1]
        depth = x[:, 1:2]
        alpha = self.to_alpha(x)
        beta = self.to_beta(x)
        gamma = self.to_gamma(x)
        compensation = alpha * depth**2 + beta * depth + gamma
        return reflectance + compensation


class SingleChannelNet(nn.Module):
    def __init__(self, use_aspp: bool = True):
        super().__init__()
        self.use_aspp = use_aspp
        self.conv1_d = conv_bn_relu(1, 64, kernel_size=3, stride=1, padding=1)
        self.AFM1 = ASPP_w_deform(2, 2, atrous_rates=[1, 6, 12, 18]) if use_aspp else None
        pretrained_model = resnet.__dict__['resnet34'](weights=None)
        pretrained_model.apply(init_weights)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model
        self.conv6 = conv_bn_relu(512, 512, kernel_size=3, stride=2, padding=1)
        self.AFM2 = ASPP_w_deform(512, 512, atrous_rates=[1, 6, 12, 18]) if use_aspp else None
        ks, st = 3, 2
        self.convt5 = convt_bn_relu(512, 256, kernel_size=ks, stride=st, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(768, 128, kernel_size=ks, stride=st, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(256 + 128, 64, kernel_size=ks, stride=st, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(128 + 64, 64, kernel_size=ks, stride=st, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(128, 64, kernel_size=ks, stride=1, padding=1)
        self.convtf = conv_bn_relu(128, 1, kernel_size=1, stride=1, bn=False, relu=False)

    def forward(self, x):
        conv1 = self.conv1_d(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        convt5 = self.convt5(conv6)
        if convt5.shape[2:] != conv5.shape[2:]:
            convt5 = F.interpolate(convt5, size=conv5.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat((convt5, conv5), dim=1)
        convt4 = self.convt4(y)
        if convt4.shape[2:] != conv4.shape[2:]:
            convt4 = F.interpolate(convt4, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat((convt4, conv4), dim=1)
        convt3 = self.convt3(y)
        if convt3.shape[2:] != conv3.shape[2:]:
            convt3 = F.interpolate(convt3, size=conv3.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat((convt3, conv3), dim=1)
        convt2 = self.convt2(y)
        if convt2.shape[2:] != conv2.shape[2:]:
            convt2 = F.interpolate(convt2, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat((convt2, conv2), dim=1)
        convt1 = self.convt1(y)
        if convt1.shape[2:] != conv1.shape[2:]:
            convt1 = F.interpolate(convt1, size=conv1.shape[2:], mode='bilinear', align_corners=False)
        y = torch.cat((convt1, conv1), dim=1)
        return self.convtf(y)


class DualChannelNet(nn.Module):
    def __init__(self, use_aspp: bool = True):
        super().__init__()
        self.use_aspp = use_aspp
        self.conv1_d = conv_bn_relu(2, 64, kernel_size=3, stride=1, padding=1)
        self.AFM1 = ASPP_w_deform(2, 2, atrous_rates=[1, 6, 12, 18]) if use_aspp else None
        pretrained_model = resnet.__dict__['resnet34'](weights=None)
        pretrained_model.apply(init_weights)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model
        self.conv6 = conv_bn_relu(512, 512, kernel_size=3, stride=2, padding=1)
        self.AFM2 = ASPP_w_deform(512, 512, atrous_rates=[1, 6, 12, 18]) if use_aspp else None
        ks, st = 3, 2
        self.convt5 = convt_bn_relu(512, 256, kernel_size=ks, stride=st, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(768, 128, kernel_size=ks, stride=st, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(256 + 128, 64, kernel_size=ks, stride=st, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(128 + 64, 64, kernel_size=ks, stride=st, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(128, 64, kernel_size=ks, stride=1, padding=1)
        self.convtf = conv_bn_relu(128, 2, kernel_size=1, stride=1, bn=False, relu=False)
        self.comp = DisCompensation()

    def forward(self, x):
        conv1 = self.conv1_d(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)
        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)
        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)
        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)
        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)
        out = self.convtf(y)
        out2 = self.comp(out)
        return out, out2


def build_net(view_type, use_aspp: bool = True):
    if view_type == "virtual_camera":
        return SingleChannelNet(use_aspp=use_aspp)
    return DualChannelNet(use_aspp=use_aspp)
