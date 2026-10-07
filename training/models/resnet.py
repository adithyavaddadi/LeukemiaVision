import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 with transfer learning and dual-output feature map support.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.3):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(base.fc.in_features, num_classes)
        self._target_layer = self.layer4[-1].conv3

    def forward(self, x, return_feature_maps=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f = self.layer4(x)  # (B, 2048, 7, 7)

        pooled = self.avgpool(f)
        flattened = torch.flatten(pooled, 1)
        dropped = self.dropout(flattened)
        logits = self.fc(dropped)

        if return_feature_maps:
            return logits, f
        return logits

    def get_last_conv_layer(self):
        return self._target_layer

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
