"""Image and text embedding encoders for the experiment runner."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 512, pretrained: bool = True, normalize: bool = True, image_channels: int = 3, image_size: int = 224, pretrained_weights: str | None = None, weights_path: str | None = None):
        super().__init__()
        self.normalize = normalize
        self.input_channels = int(image_channels)
        if backbone == "resnet50":
            weights = getattr(models.ResNet50_Weights, pretrained_weights or "DEFAULT") if pretrained else None
            if weights_path:
                weights = None
            network = models.resnet50(weights=weights)
            if weights_path:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                network.load_state_dict(state.get("state_dict", state), strict=True)
            feature_dim = network.fc.in_features
            network.fc = nn.Identity()
            self.network = network
        elif backbone == "resnet18":
            weights = getattr(models.ResNet18_Weights, pretrained_weights or "DEFAULT") if pretrained else None
            if weights_path:
                weights = None
            network = models.resnet18(weights=weights)
            if weights_path:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                network.load_state_dict(state.get("state_dict", state), strict=True)
            feature_dim = network.fc.in_features
            network.fc = nn.Identity()
            self.network = network
        elif backbone == "vit_b_16":
            weights = getattr(models.ViT_B_16_Weights, pretrained_weights or "DEFAULT") if pretrained else None
            if weights_path:
                weights = None
            network = models.vit_b_16(weights=weights)
            if weights_path:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                network.load_state_dict(state.get("state_dict", state), strict=True)
            feature_dim = network.heads.head.in_features
            network.heads = nn.Identity()
            self.network = network
        elif backbone == "kor_cnn":
            self.network = SmallGlyphBackbone(in_channels=self.input_channels)
            feature_dim = self.network.output_dim
        elif backbone == "kornet2":
            import json
            from pathlib import Path
            from configuration import configure_model

            config_path = Path(__file__).resolve().parents[1] / "zoo" / "KorNet2.json"
            arch = json.loads(config_path.read_text(encoding="utf-8"))
            feature_dim = int(arch["layers"]["output"]["params"]["out"])
            self.network = configure_model(arch, [self.input_channels, image_size, image_size])
        else:
            raise ValueError(f"Unknown image backbone {backbone!r}")
        self.projection = nn.Linear(feature_dim, embedding_dim) if feature_dim != embedding_dim else nn.Identity()
        self.output_dim = embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.network(images)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1) if self.normalize else embeddings


class SmallGlyphBackbone(nn.Module):
    """Compact 37x37 grayscale encoder for OCR-specific ablations."""

    output_dim = 256

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Softsign(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.Softsign(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Softsign(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.Softsign(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.Softsign(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output = nn.Linear(128, self.output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(images).flatten(1))


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "FacebookAI/roberta-base",
        embedding_dim: int = 768,
        pooling: str = "mean",
        normalize: bool = True,
        local_files_only: bool = False,
        freeze_backbone: bool = False,
        revision: str | None = None,
    ):
        super().__init__()
        from transformers import AutoModel

        if pooling not in ("mean", "cls"):
            raise ValueError("Text pooling must be 'mean' or 'cls'")
        self.backbone = AutoModel.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
        self.pooling = pooling
        self.normalize = normalize
        hidden = int(self.backbone.config.hidden_size)
        self.projection = nn.Linear(hidden, embedding_dim) if hidden != embedding_dim else nn.Identity()
        self.output_dim = embedding_dim
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(self, token_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.backbone(**token_batch)
        hidden = output.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = token_batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embeddings = self.projection(pooled)
        return F.normalize(embeddings, dim=-1) if self.normalize else embeddings


def build_encoder(config: dict) -> nn.Module:
    modality = config.get("modality", "image")
    if modality == "image":
        return ImageEncoder(
            backbone=config.get("backbone", "resnet50"),
            embedding_dim=int(config.get("embedding_dim", 512)),
            pretrained=bool(config.get("pretrained", True)),
            normalize=bool(config.get("normalize", True)),
            image_channels=int(config.get("image_channels", 3)),
            image_size=int(config.get("image_size", 224)),
            pretrained_weights=config.get("pretrained_weights"),
            weights_path=config.get("weights_path"),
        )
    if modality == "text":
        return TextEncoder(
            model_name=config.get("model_name", "FacebookAI/roberta-base"),
            embedding_dim=int(config.get("embedding_dim", 768)),
            pooling=config.get("pooling", "mean"),
            normalize=bool(config.get("normalize", True)),
            local_files_only=bool(config.get("local_files_only", False)),
            freeze_backbone=bool(config.get("freeze_backbone", False)),
            revision=config.get("revision"),
        )
    raise ValueError(f"Unsupported modality {modality!r}")

