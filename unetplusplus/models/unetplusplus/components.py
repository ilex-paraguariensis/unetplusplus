from torch import nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.downsample = in_channels != out_channels

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if self.downsample:
            self.conv = nn.Sequential(
                ResidualBlock(out_channels), ResidualBlock(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                ResidualBlock(in_channels), ResidualBlock(in_channels)
            )

    def forward(self, x):
        if self.downsample:
            x = self.down(x)
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, num_x=2, num_conv=2):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        conv = []
        conv.append(
            nn.Sequential(
                nn.Conv2d(in_channels * num_x, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )
        )

        for _ in range(num_conv):
            conv.append(ResidualBlock(in_channels))
        self.conv = nn.Sequential(*conv)

    def forward(self, x_up, *xs):
        xs = list(xs)
        xs.append(self.up(x_up))
        x = torch.cat(xs, dim=1)
        return self.conv(x)
