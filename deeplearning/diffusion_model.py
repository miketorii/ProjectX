import torch
from torch import nn

def _pos_encoding(t, output_dim, device="cpu"):
    D = output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D,device=device)
    div_term = 10000 ** (i/D)

    v[0::2] = torch.sin(t/div_term[0::2])
    v[1::2] = torch.cos(t/div_term[1::2])

    return v

def pos_encoding(ts, output_dim, device="cpu"):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)
 
    return v

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N,C,1,1)
        y = self.convs(x, v)
        return y


class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128+256, 128, time_embed_dim)
        self.up1 = ConvBlock(128+64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        print("UNet in", x.shape)
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        print("UNet down1", x.shape)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)
        print("UNet down2", x.shape)

        x = self.bot1(x, v)
        print("UNet bot1", x.shape)

        x = self.upsample(x)
        x = torch.cat([x,x2], dim=1)
        x = self.up2(x, v)
        print("UNet up2", x.shape)
        x = self.upsample(x)
        x = torch.cat([x,x1], dim=1)
        x = self.up1(x, v)
        print("UNet up1", x.shape)

        x = self.out(x)

        return x

'''    
if __name__ == "__main__":
    print("---start---")
    v = pos_encoding(torch.tensor([1,2,3]), 16)
    print(v.shape, "\n", v)
'''

if __name__ == "__main__":
    print("---start---")

    model = UNet()
    print(model)

    x = torch.randn(10, 1, 28, 28)
    timesteps = torch.tensor([0,1,2,3,4,5,6,7,8.9,10])
    y = model(x, timesteps)

#    print(y.shape)
