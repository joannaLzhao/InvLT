import torch
import torch.nn as nn
import torch.nn.functional as F


# Patch Embedding

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


# Transformer Encoder Block

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h

        return x


# Vision Transformer

class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        num_classes=10,
        temp=1.0,
        drop_rate=0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, drop=drop_rate)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        self.temp = temp

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        x = self.norm(x)
        cls = x[:, 0]

        out = self.fc(cls) / self.temp
        return out


def vit_b_16(temp=1.0, **kwargs):
    return ViT(
        img_size=32,
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        temp=temp,
        **kwargs
    )

def vit_l_16(temp=1.0, **kwargs):
    return ViT(
        img_size=32,
        patch_size=4,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        temp=temp,
        **kwargs
    )

def vit_h_14(temp=1.0, **kwargs):
    return ViT(
        img_size=32,
        patch_size=4,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        temp=temp,
        **kwargs
    )
