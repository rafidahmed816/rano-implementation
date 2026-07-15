"""HiFi-GAN vocoder (Kong et al., 2020) — Generator + MPD/MSD discriminators + losses.

Trains to invert THIS project's mel (torchaudio MelSpectrogram, 22050 Hz, n_fft 1024,
hop 256, 80 mels, htk/power-2, log) back to waveform. Product of upsample rates = hop
(8*8*2*2 = 256). Use with train_vocoder.py; infer with `Generator.remove_wn()` + forward.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

LRELU = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(k, d=1):
    return int((k * d - d) / 2)


class ResBlock(nn.Module):
    def __init__(self, ch, k=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(ch, ch, k, 1, dilation=d, padding=get_padding(k, d))) for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(ch, ch, k, 1, dilation=1, padding=get_padding(k, 1))) for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
            x = xt + x
        return x

    def remove_wn(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(nn.Module):
    def __init__(self, n_mels=80, upsample_rates=(8, 8, 2, 2),
                 upsample_kernel_sizes=(16, 16, 4, 4), upsample_initial_channel=512,
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(n_mels, upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(
                upsample_initial_channel // (2 ** i),
                upsample_initial_channel // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU))
            xs = None
            for j in range(self.num_kernels):
                blk = self.resblocks[i * self.num_kernels + j]
                xs = blk(x) if xs is None else xs + blk(x)
            x = xs / self.num_kernels
        x = torch.tanh(self.conv_post(F.leaky_relu(x, LRELU)))
        return x  # (B, 1, T)

    def remove_wn(self):
        for l in self.ups:
            remove_weight_norm(l)
        for b in self.resblocks:
            b.remove_wn()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(nn.Module):
    def __init__(self, period, k=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1, 32, (k, 1), (stride, 1), padding=(get_padding(k), 0))),
            weight_norm(Conv2d(32, 128, (k, 1), (stride, 1), padding=(get_padding(k), 0))),
            weight_norm(Conv2d(128, 512, (k, 1), (stride, 1), padding=(get_padding(k), 0))),
            weight_norm(Conv2d(512, 1024, (k, 1), (stride, 1), padding=(get_padding(k), 0))),
            weight_norm(Conv2d(1024, 1024, (k, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect")
            t = x.shape[2]
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(Conv1d(1, 128, 15, 1, padding=7)),
            norm(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        yr, yg, fr, fg = [], [], [], []
        for d in self.discriminators:
            a, fa = d(y); b, fb = d(y_hat)
            yr.append(a); yg.append(b); fr.append(fa); fg.append(fb)
        return yr, yg, fr, fg


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        yr, yg, fr, fg = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y); y_hat = self.meanpools[i - 1](y_hat)
            a, fa = d(y); b, fb = d(y_hat)
            yr.append(a); yg.append(b); fr.append(fa); fg.append(fb)
        return yr, yg, fr, fg


# ---- losses ----
def feature_loss(fmap_r, fmap_g):
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for r, g in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(r - g))
    return loss * 2.0


def discriminator_loss(d_real, d_gen):
    loss = 0.0
    for dr, dg in zip(d_real, d_gen):
        loss = loss + torch.mean((1 - dr) ** 2) + torch.mean(dg ** 2)
    return loss


def generator_loss(d_gen):
    loss = 0.0
    for dg in d_gen:
        loss = loss + torch.mean((1 - dg) ** 2)
    return loss
