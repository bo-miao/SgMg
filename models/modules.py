import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
from einops import rearrange


class LFMResizeAdaptive(nn.Module):
    def __init__(self, num_channels, sigma):
        super(LFMResizeAdaptive, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.sigma = sigma

        self.laplace = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7, device='cpu'):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

        yv = yv.unsqueeze(0).float().to(device)
        xv = xv.unsqueeze(0).float().to(device)
        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))
        return g.unsqueeze(0)

    def forward(self, x, gauss_map=None):
        b, c, h, w = x.shape
        x = x.float()

        # compute coef for gaussian 0~1
        coef = self.laplace(x)
        coef = self.fc(self.pool(coef).view(b, c)).view(b, 1, 1, 1)

        y = torch.fft.fft2(x)

        h_idx, w_idx = h // 2, w // 2
        if gauss_map is None:
            high_filter = self.make_gaussian(h_idx, w_idx, h, w,  self.sigma, device=x.device)
        else:
            high_filter = F.interpolate(gauss_map, size=(h, w), mode='bilinear', align_corners=False)

        y = y * (1 - coef * high_filter)

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = F.relu(self.conv1(y_f))

        y = self.conv2(y).float()
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)

        y = torch.fft.ifft2(y, s=(h, w)).float()
        return x + y, high_filter


if __name__ == "__main__":
    model = LFMResizeAdaptive(256, 3)
    data = torch.rand(2,256,8,8)
    res = model(data)
