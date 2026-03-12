from __future__ import annotations

import torch
import torch.nn as nn


class REDCNN(nn.Module):
    def __init__(self, base_channels: int = 96, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)

        self.deconv1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.deconv2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.deconv3 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.deconv4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.deconv5 = nn.ConvTranspose2d(base_channels, 1, kernel_size=kernel_size, padding=padding)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))

        y1 = self.relu(self.deconv1(x5) + x4)
        y2 = self.relu(self.deconv2(y1) + x3)
        y3 = self.relu(self.deconv3(y2) + x2)
        y4 = self.relu(self.deconv4(y3) + x1)
        return self.deconv5(y4) + x
