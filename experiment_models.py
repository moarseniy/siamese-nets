"""Image and text embedding encoders for the experiment runner."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path


_TORCHVISION_WEIGHT_ENUMS = {
    "resnet50": models.ResNet50_Weights,
    "resnet18": models.ResNet18_Weights,
    "vit_b_16": models.ViT_B_16_Weights,
    "mobilenet_v2": models.MobileNet_V2_Weights,
}


def configure_model_cache(model_config: dict, models_dir: str | Path | None) -> dict:
    """Resolve pretrained assets from a user-supplied local models directory."""
    resolved = dict(model_config)
    if models_dir is None:
        return resolved

    root = Path(models_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Models directory does not exist: {root}")

    modality = resolved.get("modality", "image")
    if modality == "text":
        model_id = str(resolved.get("model_name", "FacebookAI/roberta-base"))
        parts = model_id.split("/")
        candidates = [
            root.joinpath(*parts),
            root / model_id.replace("/", "--"),
            root / f"models--{model_id.replace('/', '--')}",
            root / parts[-1],
            root,
        ]
        local_path = None
        for candidate in candidates:
            if (candidate / "config.json").is_file():
                local_path = candidate
                break
            snapshots = candidate / "snapshots"
            if snapshots.is_dir():
                pinned_snapshot = snapshots / str(resolved.get("revision", ""))
                if (pinned_snapshot / "config.json").is_file():
                    local_path = pinned_snapshot
                    break
        if local_path is None:
            expected = root.joinpath(*parts)
            raise FileNotFoundError(
                f"Local Hugging Face model {model_id!r} was not found under {root}. "
                f"Expected a model folder such as {expected} containing config.json and weights."
            )
        resolved["local_model_path"] = str(local_path.resolve())
        resolved["local_files_only"] = True

    elif modality == "image":
        # Keep TorchVision's cache under the caller-owned root. We resolve and
        # load the expected file explicitly below, so this mode never falls
        # back to a network download.
        torch_hub_dir = root / "torchvision"
        torch.hub.set_dir(str(torch_hub_dir))
        weights_path = resolved.get("weights_path")
        if weights_path:
            local_weights = Path(weights_path).expanduser()
            if not local_weights.is_absolute():
                local_weights = root / local_weights
            if not local_weights.is_file():
                raise FileNotFoundError(f"Local model weights do not exist: {local_weights}")
            resolved["weights_path"] = str(local_weights.resolve())
        elif resolved.get("pretrained", True) and resolved.get("backbone") in _TORCHVISION_WEIGHT_ENUMS:
            weight_enum = _TORCHVISION_WEIGHT_ENUMS[resolved["backbone"]]
            weight_name = resolved.get("pretrained_weights") or "DEFAULT"
            weights = getattr(weight_enum, weight_name)
            filename = Path(weights.url).name
            local_weights = torch_hub_dir / "checkpoints" / filename
            if not local_weights.is_file():
                raise FileNotFoundError(
                    f"Local TorchVision weights for {resolved['backbone']} ({weight_name}) were not found: "
                    f"{local_weights}"
                )
            resolved["weights_path"] = str(local_weights.resolve())

    return resolved


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

            config_path = Path(__file__).resolve().parent / "zoo" / "KorNet2.json"
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
        local_model_path: str | None = None,
    ):
        super().__init__()
        from transformers import AutoModel

        if pooling not in ("mean", "cls"):
            raise ValueError("Text pooling must be 'mean' or 'cls'")
        source = local_model_path or model_name
        self.backbone = AutoModel.from_pretrained(
            source,
            revision=None if local_model_path else revision,
            local_files_only=bool(local_files_only or local_model_path),
        )
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
            local_model_path=config.get("local_model_path"),
        )
    raise ValueError(f"Unsupported modality {modality!r}")

