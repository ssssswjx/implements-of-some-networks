import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        print(x.shape)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        pre = x
        x = self.down(x)
        return x, pre


class UPsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, pre_x, interpolate=True):
        x = self.up(x)

        if (x.size(2) != pre_x.size(2)) or (x.size(3) != pre_x.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=(pre_x.size(2), pre_x.size(3)),
                                  mode="bilinear", align_corners=True)
        print(x.shape)
        print(pre_x.shape)
        x = torch.cat((x, pre_x), dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.encoder = nn.ModuleList([
            Downsample(in_channels, 64),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512),
        ])
        self.bottleneck = DoubleConv(512, 1024)
        self.decoder = nn.ModuleList([
            UPsample(1024, 512),
            UPsample(512, 256),
            UPsample(256, 128),
            UPsample(128, 64),
        ])
        self.finalneck = nn.Conv2d(64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        add = []
        for layer in self.encoder:
            x, temp = layer(x)
            add.append(temp)
        x = self.bottleneck(x)
        add = add[::-1]
        for idx, layer in enumerate(self.decoder):
            x = layer(x, add[idx])
        x = self.finalneck(x)
        return x


def test():
    data = torch.randn((3, 3, 572, 572))
    net = Unet(in_channels=3, num_classes=1)
    pred = net(data)
    print(data.shape)
    print(pred.shape)


if __name__ == '__main__':
    test()
