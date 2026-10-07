import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.block(x)

class BaselineCNN(nn.Module):
    """
    4-Block Convolutional Neural Network Baseline.
    Equipped with Global Average Pooling (GAP) for analytical Grad-CAM support.
    """
    def __init__(self, num_classes=2, dropout_p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 224 -> 112
            ConvBlock(32, 64),   # 112 -> 56
            ConvBlock(64, 128),  # 56 -> 28
            ConvBlock(128, 256)  # 28 -> 14
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(256, num_classes)
        self._target_layer = self.features[3].block[0]  # Last conv layer for Grad-CAM

    def forward(self, x, return_feature_maps=False):
        f = self.features(x)
        pooled = self.gap(f)
        flattened = torch.flatten(pooled, 1)
        dropped = self.dropout(flattened)
        logits = self.classifier(dropped)
        if return_feature_maps:
            return logits, f
        return logits

    def get_last_conv_layer(self):
        return self._target_layer
