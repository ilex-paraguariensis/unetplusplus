from torch import nn
from .components import EncoderBlock, DecoderBlock
import ipdb


class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        channels=(32, 64, 128, 256, 512),
        average=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.average = average
        self.in_conv = nn.Sequential(
            nn.Conv2d(n_channels, channels[0] // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[0] // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[0] // 2, channels[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block00 = EncoderBlock(channels[0], channels[0])
        self.block10 = EncoderBlock(channels[0], channels[1])
        self.block20 = EncoderBlock(channels[1], channels[2])
        self.block30 = EncoderBlock(channels[2], channels[3])
        self.block40 = EncoderBlock(channels[3], channels[4])

        self.block01 = DecoderBlock(channels[0], num_x=2)
        self.block02 = DecoderBlock(channels[0], num_x=3)
        self.block03 = DecoderBlock(channels[0], num_x=4)
        self.block04 = DecoderBlock(channels[0], num_x=5)

        self.block11 = DecoderBlock(channels[1], num_x=2)
        self.block12 = DecoderBlock(channels[1], num_x=3)
        self.block13 = DecoderBlock(channels[1], num_x=4)

        self.block21 = DecoderBlock(channels[2], num_x=2)
        self.block22 = DecoderBlock(channels[2], num_x=3)

        self.block31 = DecoderBlock(channels[3], num_x=2)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[0] // 2, 2, stride=2),
            nn.BatchNorm2d(channels[0] // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(channels[0] // 2, n_classes, (2, 2), stride=2),
            nn.BatchNorm2d(n_classes),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, in_x, out_mode="L4"):
        x = self.in_conv(in_x)

        out00 = self.block00(x)
        out10 = self.block10(out00)
        out01 = self.block01(out10, out00)
        if out_mode == "L1":
            return self.out_conv(out01)

        out20 = self.block20(out10)
        out11 = self.block11(out20, out10)
        out02 = self.block02(out11, out00, out01)
        if out_mode == "L2":
            if self.average:
                return self.out_conv((out01 + out02) / 2)
            else:
                return self.out_conv(out02)

        out30 = self.block30(out20)
        out21 = self.block21(out30, out20)
        out12 = self.block12(out21, out10, out11)
        out03 = self.block03(out12, out00, out01, out02)
        if out_mode == "L3":
            if self.average:
                return self.out_conv((out01 + out02 + out03) / 3)
            else:
                return self.out_conv(out03)

        out40 = self.block40(out30)
        out31 = self.block31(out40, out30)
        out22 = self.block22(out31, out20, out21)
        out13 = self.block13(out22, out10, out11, out12)
        out04 = self.block04(out13, out00, out01, out02, out03)
        if self.average:
            return self.out_conv((out01 + out02 + out03 + out04) / 4)[
                :, :, : in_x.shape[-2], : in_x.shape[-1]
            ]
        else:
            return self.out_conv(out04)[:, :, : in_x.shape[-2], : in_x.shape[-1]]
