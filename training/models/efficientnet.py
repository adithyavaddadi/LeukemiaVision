import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with ImageNet pretraining, GAP, and dual-output feature maps.
    Top candidate for ONNX export and CPU serverless inference.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_p=0.2):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)
        
        self.features = base.features
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=dropout_p)
        in_features = base.classifier[1].in_features
        self.classifier = nn.Linear(in_features, num_classes)
        self._target_layer = self.features[-1]  # Conv2dNormActivation at top of features

    def forward(self, x, return_feature_maps=False):
        f = self.features(x)  # (B, 1280, 7, 7)
        pooled = self.avgpool(f)
        flattened = torch.flatten(pooled, 1)
        dropped = self.dropout(flattened)
        logits = self.classifier(dropped)

        if return_feature_maps:
            return logits, f
        return logits

    def get_last_conv_layer(self):
        return self._target_layer

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
