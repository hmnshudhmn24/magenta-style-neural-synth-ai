import torch, torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        h = self.net(x)
        z = self.proj(h)
        return z

class Decoder(nn.Module):
    def __init__(self, out_ch=1, latent_dim=128, start_hw=(5,8)):
        super().__init__()
        C = 256
        self.fc = nn.Sequential(nn.Linear(latent_dim, C*start_hw[0]*start_hw[1]),
                                nn.LeakyReLU(0.1))
        self.start_hw = start_hw
        self.net = nn.Sequential(
            nn.ConvTranspose2d(C, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, self.start_hw[0], self.start_hw[1])
        x = self.net(h)
        return x

class TimbreAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc = Encoder(latent_dim=latent_dim)
        self.dec = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z
