import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from flpnet.utils import Encoder_block, BasicBlock


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block=1, scale_factor=None, block=BasicBlock):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        block_list = []
        block_list.append(block(2 * out_ch, out_ch))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out


class MlpDecoderHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 128, 256, 512]):
        super(MlpDecoderHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels, c5_in_channels = in_channels

        self.linear_c4 = up_block(in_ch=c5_in_channels, out_ch=c4_in_channels, scale_factor=2)
        self.linear_c3 = up_block(in_ch=c4_in_channels, out_ch=c3_in_channels, scale_factor=2)
        self.linear_c2 = up_block(in_ch=c3_in_channels, out_ch=c2_in_channels, scale_factor=2)
        self.linear_c1_ = up_block(in_ch=c2_in_channels, out_ch=c1_in_channels, scale_factor=2)
        self.linear_c1 = up_block(in_ch=c1_in_channels, out_ch=c1_in_channels, scale_factor=2)

        self.linear_pred = nn.Conv2d(c1_in_channels, num_classes, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(c1_in_channels, num_classes, kernel_size=1)
        self.linear_pred_2 = nn.Conv2d(c2_in_channels, num_classes, kernel_size=1)
        self.linear_pred_3 = nn.Conv2d(c3_in_channels, num_classes, kernel_size=1)
        self.linear_pred_4 = nn.Conv2d(c4_in_channels, num_classes, kernel_size=1)
        self.linear_pred_5 = nn.Conv2d(c5_in_channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c1_, c2, c3, c4, c5 = inputs

        _c4 = self.linear_c4(c5, c4)
        _c3 = self.linear_c3(_c4, c3)
        _c2 = self.linear_c2(_c3, c2)
        _c2_ = self.linear_c1_(_c2, c1_)
        _c1 = self.linear_c1(_c2_, c1)

        x = self.linear_pred(_c1)
        return x

class FLPNet(nn.Module):

    def __init__(self, in_ch=1, num_classes=21):
        super(FLPNet, self).__init__()
        self.in_channels = [32, 64, 128, 256, 512]

        self.backbone = Encoder_block(in_chans=in_ch, embed_dims=[32, 64, 128, 256, 512], num_heads=[1, 2, 4, 8],
                                       mlp_ratios=[4, 4, 4, 4],
                                       qkv_bias=True, depths=[3, 4, 6, 3],
                                       sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

        self.embedding_dim = 512
        self.decode_head = MlpDecoderHead(num_classes, self.in_channels)

    def forward(self, inputs):
        x = self.backbone.forward(inputs)

        x = self.decode_head.forward(x)

        return x
