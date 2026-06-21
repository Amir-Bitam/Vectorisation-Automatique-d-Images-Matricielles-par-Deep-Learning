"""DINOv3 patch-token encoder and attention stroke head."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import timm


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Stroke queries attend to DINO patch-token keys/values."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, stroke_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(stroke_tokens)
        context = self.context_norm(patch_tokens)
        attended, _ = self.attn(query=query, key=context, value=context, need_weights=False)
        stroke_tokens = stroke_tokens + attended
        stroke_tokens = stroke_tokens + self.ffn(self.ffn_norm(stroke_tokens))
        return stroke_tokens


class SelfAttentionBlock(nn.Module):
    """One interaction block between predicted strokes."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DinoV3StrokeEncoder(nn.Module):
    """Predicts normalized closed cubic Bezier paths from one RGB crop.

    Output shape is ``[batch, num_paths, 27]``. Values 0:24 are 12 normalized
    ``(x, y)`` points and values 24:27 are RGB fill values. Sigmoid constrains
    every value to ``[0, 1]``.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3",
        pretrained: bool = True,
        num_paths: int = 32,
        output_dim: int = 27,
        image_size: int = 224,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.output_dim = output_dim
        self.normalize_input = normalize_input

        self.dino = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=image_size)
        embed_dim = int(getattr(self.dino, "embed_dim"))
        prefix_tokens = int(getattr(self.dino, "num_prefix_tokens", 1))
        self.prefix_tokens = prefix_tokens

        heads = min(num_heads, embed_dim)
        while embed_dim % heads != 0 and heads > 1:
            heads -= 1

        self.stroke_tokens = nn.Parameter(torch.zeros(1, num_paths, embed_dim))
        self.cross_attention = CrossAttentionBlock(embed_dim, num_heads=heads, mlp_ratio=mlp_ratio)
        self.self_attention = SelfAttentionBlock(embed_dim, num_heads=heads, mlp_ratio=mlp_ratio)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.linear_head = nn.Linear(embed_dim, output_dim)

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return image
        return (image - self.image_mean.to(image.dtype)) / self.image_std.to(image.dtype)

    def extract_patch_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        image = self._normalize(image)

        features = self.dino.forward_features(image)
        if isinstance(features, dict):
            features = features.get("x", features.get("last_hidden_state"))
        if features is None or features.ndim != 3:
            raise RuntimeError("DINOv3 forward_features must return a [B, tokens, C] tensor.")

        # Timm DINOv3 EVA models expose CLS plus register prefix tokens. Drop all
        # prefix tokens so only the 14x14 image patch tokens remain.
        patch_tokens = features[:, self.prefix_tokens :, :]
        if patch_tokens.shape[1] != 196:
            if features.shape[1] == 197:
                patch_tokens = features[:, 1:, :]
            else:
                raise RuntimeError(
                    f"Expected 196 patch tokens at 224x224, got {patch_tokens.shape[1]} "
                    f"from {features.shape[1]} total tokens."
                )
        return patch_tokens

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.extract_patch_tokens(image)
        stroke_tokens = self.stroke_tokens.expand(image.shape[0], -1, -1)
        stroke_tokens = self.cross_attention(stroke_tokens, patch_tokens)
        stroke_tokens = self.self_attention(stroke_tokens)
        raw = self.linear_head(self.head_norm(stroke_tokens))
        return torch.sigmoid(raw)
