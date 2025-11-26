# models.py
import torch.nn as nn
import torch


# ---------- helpers ----------

def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


# ---------- generator ----------

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    9-block ResNet generator for 256x256 images.
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        model = []

        # c7s1-64
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        # downsample: 64->128->256
        in_c = ngf
        out_c = in_c * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(True),
            ]
            in_c = out_c
            out_c *= 2

        # 9 ResNet blocks at 256 channels
        for _ in range(n_blocks):
            model += [ResnetBlock(in_c)]

        # upsample: 256->128->64
        out_c = in_c // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(True),
            ]
            in_c = out_c
            out_c //= 2

        # final conv
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, output_nc, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ---------- discriminator ----------

class NLayerDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
