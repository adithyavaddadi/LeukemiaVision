from .cnn import BaselineCNN
from .resnet import ResNet50Classifier
from .efficientnet import EfficientNetB0Classifier
from .transformer import VisionTransformer
from .cnn_transformer import HybridCNNTransformer

def build_model(model_name, num_classes=2, pretrained=True, dropout=0.3):
    name = model_name.lower().replace("-", "_")
    if name in ["baseline_cnn", "cnn"]:
        return BaselineCNN(num_classes=num_classes, dropout_p=dropout)
    elif name in ["resnet50", "resnet"]:
        return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout)
    elif name in ["efficientnet_b0", "efficientnet"]:
        return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout)
    elif name in ["transformer", "vit"]:
        return VisionTransformer(num_classes=num_classes, dropout_p=dropout)
    elif name in ["cnn_transformer", "hybrid"]:
        return HybridCNNTransformer(num_classes=num_classes, dropout_p=dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Options: baseline_cnn, resnet50, efficientnet_b0, transformer, cnn_transformer")
