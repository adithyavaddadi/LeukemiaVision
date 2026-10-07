import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Splits an image into fixed-size patches and projects them into an embedding space."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    Standalone Vision Transformer (ViT-style) for blood smear classification.
    Splits input image into 16x16 patches, projects to tokens, adds learnable
    [CLS] token + positional embeddings, then processes through Transformer
    encoder layers with Multi-Head Self-Attention.

    Architecture:
        Input (224x224) -> 16x16 Patch Embedding -> 196 tokens
        -> [CLS] token prepend -> 197 tokens + Positional Embedding
        -> L Transformer Encoder Layers (MHSA + MLP)
        -> [CLS] output -> LayerNorm -> Linear classifier
    """
    def __init__(self, num_classes=2, img_size=224, patch_size=16, embed_dim=256,
                 num_heads=4, num_layers=4, mlp_ratio=2, dropout_p=0.2):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196 for 224/16

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout_p)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout_p,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # For Grad-CAM: the patch embedding conv acts as the "last conv layer"
        self._target_layer = self.patch_embed.proj

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, return_feature_maps=False):
        B = x.shape[0]

        # 1. Patch embedding (acts as the feature extractor)
        patch_features = self.patch_embed.proj(x)  # (B, embed_dim, H/P, W/P) = (B, 256, 14, 14)
        tokens = patch_features.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)

        # 2. Prepend [CLS] token + positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, tokens), dim=1)  # (B, 197, embed_dim)
        x_seq = self.pos_drop(x_seq + self.pos_embed)

        # 3. Transformer Encoder
        x_seq = self.transformer(x_seq)  # (B, 197, embed_dim)

        # 4. Classify from [CLS] token
        cls_out = self.norm(x_seq[:, 0])
        logits = self.head(cls_out)

        if return_feature_maps:
            return logits, patch_features  # Return the spatial feature map for Grad-CAM
        return logits

    def get_last_conv_layer(self):
        return self._target_layer
