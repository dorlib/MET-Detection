import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformerSegmentation(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=8, num_layers=12,
                 num_classes=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))

        # Transformer blocks for encoding the image
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])

        # Decoder layers to upsample the feature map back to the input size
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)
        )

        # Final output to get a segmentation map
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding step
        x = self.patch_embed(x)

        # Add class token and positional encoding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed

        # Pass through transformer layers
        for layer in self.encoder:
            x = layer(x)

        # We take the output of transformer encoder and reshape it for segmentation
        x = x[:, 1:].transpose(1, 2).reshape(batch_size, self.embed_dim, int(x.size(1) ** 0.5), int(x.size(1) ** 0.5))

        # Upsample to the original image size
        x = self.upsample(x)

        # Pass through decoder layers to get the segmentation output
        x = self.decoder(x)

        return x
