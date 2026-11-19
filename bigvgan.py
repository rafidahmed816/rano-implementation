"""BigVGAN-style vocoder — Snake-activated generator + Multi-Resolution Discriminator.

This is BigVGAN's CORE (SnakeBeta periodic activation + MRD), NOT the full paper:
the anti-aliased up/down-sampling filters around the activations are omitted on
purpose (fiddly to reimplement correctly; a subtle bug wastes cloud time). Snake is
the dominant quality win over HiFi-GAN, so this captures most of the benefit reliably.
For the exact anti-aliased BigVGAN, use NVIDIA's official repo + a mel patch.

Trains on THIS project's mel (audio.MelProcessor). Reuses HiFi-GAN's MPD + losses.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm

from hifigan import (get_padding, init_weights, MultiPeriodDiscriminator,
                     feature_loss, discriminator_loss, generator_loss)


class SnakeBeta(nn.Module):
    """SnakeBeta activation: x + (1/b) sin^2(a x), with per-channel log-a, log-b.
    The periodic term gives strong inductive bias for audio (BigVGAN's key idea)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))  # log-scale
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):  # (B, C, T)
        a = torch.exp(self.alpha).view(1, -1, 1)
        b = torch.exp(self.beta).view(1, -1, 1)
        return x + (1.0 / (b + 1e-9)) * torch.sin(a * x) ** 2


class AMPBlock(nn.Module):
    """Residual block with dilated convs and SnakeBeta activations."""

    def __init__(self, ch, k=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(ch, ch, k, 1, dilation=d, padding=get_padding(k, d))) for d in dilation])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(ch, ch, k, 1, dilation=1, padding=get_padding(k, 1))) for _ in dilation])
        self.acts1 = nn.ModuleList([SnakeBeta(ch) for _ in dilation])
        self.acts2 = nn.ModuleList([SnakeBeta(ch) for _ in dilation])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = c2(a2(c1(a1(x))))
            x = x + xt
        return x


class BigVGAN(nn.Module):
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
                upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, k, d))
        self.act_post = SnakeBeta(ch)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                blk = self.resblocks[i * self.num_kernels + j]
                xs = blk(x) if xs is None else xs + blk(x)
            x = xs / self.num_kernels
        x = torch.tanh(self.conv_post(self.act_post(x)))
        return x

    def remove_wn(self):
        from torch.nn.utils import remove_weight_norm
        for l in self.ups:
            remove_weight_norm(l)
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for b in self.resblocks:
            for l in b.convs1:
                remove_weight_norm(l)
            for l in b.convs2:
                remove_weight_norm(l)


class DiscriminatorR(nn.Module):
    """Spectrogram discriminator at one STFT resolution."""

    def __init__(self, resolution):
        super().__init__()
        self.n_fft, self.hop, self.win = resolution
        self.register_buffer("window", torch.hann_window(self.win))
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1, 32, (3, 9), padding=(1, 4))),
            weight_norm(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, y):  # y: (B, T)
        fmap = []
        x = torch.stft(y, self.n_fft, self.hop, self.win, window=self.window.to(y.device),
                       center=True, return_complex=True)
        x = torch.abs(x).unsqueeze(1)  # (B, 1, freq, frames)
        for l in self.convs:
            x = F.leaky_relu(l(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512))):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorR(r) for r in resolutions])

    def forward(self, y, y_hat):  # (B,1,T)
        yr, yg, fr, fg = [], [], [], []
        for d in self.discriminators:
            a, fa = d(y.squeeze(1)); b, fb = d(y_hat.squeeze(1))
            yr.append(a); yg.append(b); fr.append(fa); fg.append(fb)
        return yr, yg, fr, fg
