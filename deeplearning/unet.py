import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()

        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128+256, 128)
        self.up1 = ConvBlock(128+64, 64)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        print("UNet in", x.shape)
        x1 = self.down1(x)
        x = self.maxpool(x1)
        print("UNet down1", x.shape)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        print("UNet down2", x.shape)

        x = self.bot1(x)
        print("UNet bot1", x.shape)

        x = self.upsample(x)
        x = torch.cat([x,x2], dim=1)
        x = self.up2(x)
        print("UNet up2", x.shape)
        x = self.upsample(x)
        x = torch.cat([x,x1], dim=1)
        x = self.up1(x)
        print("UNet up1", x.shape)

        x = self.out(x)

        return x

if __name__ == "__main__":
    print("---start---")

    model = UNet()
    print(model)

    x = torch.randn(10, 1, 28, 28)
    y = model(x)

    print(y.shape)
