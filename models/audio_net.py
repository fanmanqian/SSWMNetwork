# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from models.film_layer import FilmLayer


# class Unet(nn.Module):
#     def __init__(self, n_channels=1, n_classes=32, bilinear=True):
#         super(Unet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.film_layer = FilmLayer()

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x, context=None):
#         # print(1, x.size())
#         x1 = self.inc(x)
#         # print(2, x1.size())
#         x2 = self.down1(x1)
#         # print(3, x2.size())
#         x3 = self.down2(x2)
#         # print(4, x3.size())
#         x4 = self.down3(x3)
#         # print(5, x4.size())
#         x5 = self.down4(x4)
#         # print(6, x5.size())

#         if context:
#             x5 = self.film_layer(x5, context)
#             # print(6.6, x5.size())

#         x = self.up1(x5, x4)
#         # print(5, x.size())
#         x = self.up2(x, x3)
#         # print(4, x.size())
#         x = self.up3(x, x2)
#         # print(3, x.size())
#         x = self.up4(x, x1)
#         # print(2, x.size())
#         logits = self.outc(x)
#         # print(1, logits.size())
#         return logits


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

#         self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# # class Unet(nn.Module):
# #     def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False):
# #         super(Unet, self).__init__()
# #
# #         # construct unet structure
# #         unet_block = UnetBlock(
# #             ngf * 8, ngf * 8, input_nc=None,
# #             submodule=None, innermost=True)
# #         for i in range(num_downs - 5):
# #             unet_block = UnetBlock(
# #                 ngf * 8, ngf * 8, input_nc=None,
# #                 submodule=unet_block, use_dropout=use_dropout)
# #         unet_block = UnetBlock(
# #             ngf * 4, ngf * 8, input_nc=None,
# #             submodule=unet_block)
# #         unet_block = UnetBlock(
# #             ngf * 2, ngf * 4, input_nc=None,
# #             submodule=unet_block)
# #         unet_block = UnetBlock(
# #             ngf, ngf * 2, input_nc=None,
# #             submodule=unet_block)
# #         unet_block = UnetBlock(
# #             fc_dim, ngf, input_nc=1,
# #             submodule=unet_block, outermost=True)
# #
# #         self.bn0 = nn.BatchNorm2d(1)
# #         self.unet_block = unet_block
# #
# #     def forward(self, x):
# #         x = self.bn0(x)
# #         x = self.unet_block(x)
# #         return x
# #
# #
# # # Defines the submodule with skip connection.
# # # X -------------------identity---------------------- X
# # #   |-- downsampling -- |submodule| -- upsampling --|
# # class UnetBlock(nn.Module):
# #     def __init__(self, outer_nc, inner_input_nc, input_nc=None,
# #                  submodule=None, outermost=False, innermost=False,
# #                  use_dropout=False, inner_output_nc=None, noskip=False):
# #         super(UnetBlock, self).__init__()
# #         self.outermost = outermost
# #         self.noskip = noskip
# #         self.innermost = innermost
# #         use_bias = False
# #         if input_nc is None:
# #             input_nc = outer_nc
# #         if innermost:
# #             inner_output_nc = inner_input_nc
# #         elif inner_output_nc is None:
# #             inner_output_nc = 2 * inner_input_nc
# #
# #         downrelu = nn.LeakyReLU(0.2, True)
# #         downnorm = nn.BatchNorm2d(inner_input_nc)
# #         uprelu = nn.ReLU(True)
# #         upnorm = nn.BatchNorm2d(outer_nc)
# #         upsample = nn.Upsample(
# #             scale_factor=2, mode='bilinear', align_corners=True)
# #
# #         if outermost:
# #             downconv = nn.Conv2d(
# #                 input_nc, inner_input_nc, kernel_size=4,
# #                 stride=2, padding=1, bias=use_bias)
# #             upconv = nn.Conv2d(
# #                 inner_output_nc, outer_nc, kernel_size=3, padding=1)
# #
# #             down = [downconv]
# #             up = [uprelu, upsample, upconv]
# #             model = down + [submodule] + up
# #         elif innermost:
# #             downconv = nn.Conv2d(
# #                 input_nc, inner_input_nc, kernel_size=4,
# #                 stride=2, padding=1, bias=use_bias)
# #             upconv = nn.Conv2d(
# #                 inner_output_nc, outer_nc, kernel_size=3,
# #                 padding=1, bias=use_bias)
# #
# #             down = [downrelu, downconv]
# #             up = [uprelu, upsample, upconv, upnorm]
# #             model = down + up
# #         else:
# #             downconv = nn.Conv2d(
# #                 input_nc, inner_input_nc, kernel_size=4,
# #                 stride=2, padding=1, bias=use_bias)
# #             upconv = nn.Conv2d(
# #                 inner_output_nc, outer_nc, kernel_size=3,
# #                 padding=1, bias=use_bias)
# #             down = [downrelu, downconv, downnorm]
# #             up = [uprelu, upsample, upconv, upnorm]
# #
# #             if use_dropout:
# #                 model = down + [submodule] + up + [nn.Dropout(0.5)]
# #             else:
# #                 model = down + [submodule] + up
# #
# #         self.model = nn.Sequential(*model)
# #
# #     def forward(self, x):
# #         if self.outermost or self.noskip:
# #             # print(1, x.size())
# #             # print(self.model(x).size())
# #             return self.model(x)
# #         # elif self.innermost is True:
# #         #     print(3, x.size())
# #         #     return self.model(torch.cat([x, x], 1))
# #         else:
# #             # print(2, x.size())
# #             return torch.cat([x, self.model(x)], 1)


# if __name__ == '__main__':

#     model = Unet()
#     dummy_input = torch.zeros(4, 1, 512, 320)
#     output = model(dummy_input)
#     print(output.shape)







from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

# DY-RELU
class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True),
            DyReLUB(out_ch,conv_type='2d'),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True)
            DyReLUB(out_ch,conv_type='2d'))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            #nn.ReLU(inplace=True
            DyReLUB(out_ch,conv_type='2d'))

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        # psi = DyReLUB(g1+x1,10,conv_type='2d')
        psi = self.psi(psi)
        out = x * psi
        return out
class Unet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=1, output_ch=32):
        super(Unet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out
