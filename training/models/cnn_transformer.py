import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer Architecture:
    1. CNN Backbone: ResNet-18 feature extractor capturing local texture & chromatin patterns (yields 512-ch 7x7 spatial map).
    2. Patch/Token Projection: 1x1 Conv projecting 512 channels -> embed_dim (256) and flattening 7x7 into 49 spatial tokens.
    3. Tokenization & Positional Embedding: Learnable [CLS] token prepended -> 50 sequence tokens + 1D Learnable Positional Embeddings.
    4. Transformer Encoder: L layers of Multi-Head Self-Attention (MHSA) capturing global spatial relationships across cell regions.
    5. Classification Head: MLP projecting the [CLS] token to class logits.
    """
    def __init__(self, num_classes=2, embed_dim=256, num_heads=4, num_layers=3, dropout_p=0.2):
        super().__init__()
        # CNN Backbone
        weights = ResNet18_Weights.DEFAULT
        r18 = resnet18(weights=weights)
        self.cnn_backbone = nn.Sequential(
            r18.conv1,
            r18.bn1,
            r18.relu,
            r18.maxpool,
            r18.layer1,
            r18.layer2,
            r18.layer3,
            r18.layer4  # Output shape: (B, 512, 7, 7) for 224x224 input
        )
        
        # Tokenizer: 1x1 Conv projection
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.num_patches = 7 * 7  # 49
        
        # Class token and Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout_p)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # For Grad-CAM inspection on CNN feature extractor
        self._target_layer = self.cnn_backbone[-1][-1].conv2

    def forward(self, x, return_feature_maps=False):
        B = x.shape[0]
        
        # 1. CNN Feature extraction
        f_cnn = self.cnn_backbone(x)  # (B, 512, 7, 7)
        
        # 2. Patch projection
        proj = self.proj(f_cnn)  # (B, embed_dim, 7, 7)
        tokens = proj.flatten(2).transpose(1, 2)  # (B, 49, embed_dim)
        
        # 3. Prepend [CLS] token and add positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, tokens), dim=1)  # (B, 50, embed_dim)
        x_tokens = self.pos_drop(x_tokens + self.pos_embed)
        
        # 4. Transformer Encoder
        trans_out = self.transformer(x_tokens)  # (B, 50, embed_dim)
        
        # 5. Class representation via [CLS] token
        cls_rep = self.norm(trans_out[:, 0])
        logits = self.head(cls_rep)
        
        if return_feature_maps:
            return logits, f_cnn
        return logits

    def get_last_conv_layer(self):
        return self._target_layer
