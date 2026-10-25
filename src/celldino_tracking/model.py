from __future__ import annotations

import math
from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

try:
    from trackformer.models.backbone import build_backbone
    from trackformer.util.misc import NestedTensor
except Exception:
    build_backbone = None

    class NestedTensor:
        def __init__(self, tensors: torch.Tensor, mask: torch.Tensor):
            self.tensors = tensors
            self.mask = mask

from .data import PairSample


def make_backbone_args(hidden_dim: int, backbone: str, train_backbone: bool, num_feature_levels: int) -> Namespace:
    return Namespace(
        hidden_dim=hidden_dim,
        backbone=backbone,
        lr_backbone=2e-5 if train_backbone else 0.0,
        masks=False,
        num_feature_levels=num_feature_levels,
        dilation=False,
        position_embedding="sine",
        multi_frame_attention=False,
        multi_frame_encoding=False,
    )


class SimpleResNetBackbone(nn.Module):
    def __init__(self, name: str, train_backbone: bool):
        super().__init__()
        model = getattr(torchvision.models, name)(weights=None)
        self.body = IntermediateLayerGetter(model, return_layers={"layer2": "0", "layer3": "1", "layer4": "2"})
        self.num_channels = [512, 1024, 2048] if name in {"resnet50", "resnet101"} else [128, 256, 512]
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, nested: NestedTensor):
        xs = self.body(nested.tensors)
        return [NestedTensor(x, torch.zeros(x.shape[0], x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device)) for x in xs.values()], []


def fourier_xy_time(centers: torch.Tensor, frames: torch.Tensor, num_bands: int = 16) -> torch.Tensor:
    if centers.numel() == 0:
        return centers.new_zeros((0, num_bands * 2 * 3))
    coords = torch.cat([centers, frames.float().unsqueeze(1)], dim=1)
    freqs = torch.arange(num_bands, device=centers.device, dtype=centers.dtype)
    angles = coords[..., None] * (2.0 ** freqs) * math.pi
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)


class BackboneObjectEncoder(nn.Module):
    def __init__(self, embedding_dim: int, backbone_name: str, train_backbone: bool, num_feature_levels: int):
        super().__init__()
        if build_backbone is not None:
            self.backbone = build_backbone(make_backbone_args(embedding_dim, backbone_name, train_backbone, num_feature_levels))
            feature_channels = self.backbone.num_channels[-1]
        else:
            self.backbone = SimpleResNetBackbone(backbone_name, train_backbone)
            feature_channels = self.backbone.num_channels[-1]
        self.feature_channels = feature_channels
        self.embedding_dim = embedding_dim
        self.project = nn.Sequential(
            nn.Linear(feature_channels + 7 + 96, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, sample: PairSample) -> torch.Tensor:
        if sample.centers.shape[0] == 0:
            return sample.centers.new_zeros((0, self.embedding_dim))
        mask = torch.zeros((sample.frames.shape[0], sample.frames.shape[-2], sample.frames.shape[-1]), dtype=torch.bool, device=sample.frames.device)
        features, _ = self.backbone(NestedTensor(sample.frames, mask))
        feature_map = features[-1].tensors
        object_features = sample.centers.new_zeros((sample.centers.shape[0], self.feature_channels))
        for frame_idx in (0, 1):
            keep = (sample.frame_indices == frame_idx).nonzero(as_tuple=False).flatten()
            if keep.numel() == 0:
                continue
            grid = sample.centers[keep].mul(2.0).sub(1.0).view(1, -1, 1, 2)
            sampled = F.grid_sample(feature_map[frame_idx:frame_idx + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            object_features[keep] = sampled[0, :, :, 0].transpose(0, 1)
        pos = fourier_xy_time(sample.centers, sample.frame_indices)
        return self.project(torch.cat([object_features, sample.shallow, pos], dim=1))


class CellDINOTrackingModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        heads: int = 4,
        backbone: str = "resnet50",
        train_backbone: bool = True,
        num_feature_levels: int = 3,
        max_mitosis_pairs_per_track: int = 16,
    ):
        super().__init__()
        self.encoder = BackboneObjectEncoder(embedding_dim, backbone, train_backbone, num_feature_levels)
        self.track_embedding = nn.Parameter(torch.zeros(1, embedding_dim))
        self.object_embedding = nn.Parameter(torch.zeros(1, embedding_dim))
        nn.init.normal_(self.track_embedding, std=0.02)
        nn.init.normal_(self.object_embedding, std=0.02)
        self.cross_attn = nn.MultiheadAttention(embedding_dim, heads, batch_first=True)
        self.track_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim))
        self.object_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim))
        self.division_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 1))
        self.mitosis_head = nn.Sequential(nn.LayerNorm(embedding_dim * 3), nn.Linear(embedding_dim * 3, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 1))
        self.scale = math.sqrt(float(embedding_dim))
        self.max_mitosis_pairs_per_track = max_mitosis_pairs_per_track

    def forward(self, sample: PairSample) -> Dict[str, torch.Tensor]:
        tokens = self.encoder(sample)
        prev_idx = (sample.frame_indices == 0).nonzero(as_tuple=False).flatten()
        curr_idx = (sample.frame_indices == 1).nonzero(as_tuple=False).flatten()
        if prev_idx.numel() == 0 or curr_idx.numel() == 0:
            empty = tokens.new_zeros((prev_idx.numel(), curr_idx.numel()))
            return {"tokens": tokens, "prev_idx": prev_idx, "curr_idx": curr_idx, "pair_logits": empty, "division_logits": empty.new_zeros((prev_idx.numel(),)), "mitosis_logits": empty.new_zeros((0,)), "mitosis_pairs": torch.zeros((0, 3), dtype=torch.long, device=tokens.device)}
        prev_tokens = tokens[prev_idx] + self.track_embedding
        curr_tokens = tokens[curr_idx] + self.object_embedding
        tracked, _ = self.cross_attn(prev_tokens.unsqueeze(0), curr_tokens.unsqueeze(0), curr_tokens.unsqueeze(0))
        tracked = tracked.squeeze(0)
        pair_logits = self.track_head(tracked) @ self.object_head(curr_tokens).transpose(0, 1) / self.scale
        division_logits = self.division_head(tracked).squeeze(-1)
        mitosis_pairs, mitosis_logits = self._mitosis_queries(tracked, curr_tokens, sample.centers[prev_idx], sample.centers[curr_idx])
        return {"tokens": tokens, "prev_idx": prev_idx, "curr_idx": curr_idx, "pair_logits": pair_logits, "division_logits": division_logits, "mitosis_logits": mitosis_logits, "mitosis_pairs": mitosis_pairs}

    def _mitosis_queries(self, prev_tokens: torch.Tensor, curr_tokens: torch.Tensor, prev_centers: torch.Tensor, curr_centers: torch.Tensor):
        candidates: List[torch.Tensor] = []
        pair_rows: List[List[int]] = []
        if curr_tokens.shape[0] < 2:
            return torch.zeros((0, 3), dtype=torch.long, device=curr_tokens.device), curr_tokens.new_zeros((0,))
        for p in range(prev_tokens.shape[0]):
            distances = torch.cdist(prev_centers[p:p + 1], curr_centers).flatten()
            pair_score = distances[:, None] + distances[None, :]
            triu = torch.triu_indices(curr_tokens.shape[0], curr_tokens.shape[0], offset=1, device=curr_tokens.device)
            scores = pair_score[triu[0], triu[1]]
            keep = torch.argsort(scores)[: self.max_mitosis_pairs_per_track]
            for item in keep:
                a = int(triu[0, item].item())
                b = int(triu[1, item].item())
                candidates.append(torch.cat([prev_tokens[p], curr_tokens[a], curr_tokens[b]], dim=0))
                pair_rows.append([p, a, b])
        if not candidates:
            return torch.zeros((0, 3), dtype=torch.long, device=curr_tokens.device), curr_tokens.new_zeros((0,))
        pair_tensor = torch.tensor(pair_rows, dtype=torch.long, device=curr_tokens.device)
        logits = self.mitosis_head(torch.stack(candidates, dim=0)).squeeze(-1)
        return pair_tensor, logits


class CellDINOTrackingLoss(nn.Module):
    def __init__(self, positive_weight: float = 4.0, division_weight: float = 10.0, mitosis_weight: float = 2.0):
        super().__init__()
        self.positive_weight = positive_weight
        self.division_weight = division_weight
        self.mitosis_weight = mitosis_weight

    def forward(self, outputs: Dict[str, torch.Tensor], sample: PairSample) -> Dict[str, torch.Tensor]:
        pair_logits = outputs["pair_logits"]
        if pair_logits.numel() == 0:
            zero = sample.centers.sum() * 0.0
            return {"loss": zero, "loss_pair": zero, "loss_division": zero, "loss_mitosis": zero}
        prev_ids = sample.track_ids[outputs["prev_idx"]]
        curr_ids = sample.track_ids[outputs["curr_idx"]]
        curr_parent = sample.parent_ids[outputs["curr_idx"]]
        pair_target = ((curr_ids.unsqueeze(0) == prev_ids.unsqueeze(1)) | (curr_parent.unsqueeze(0) == prev_ids.unsqueeze(1))).float()
        pair_weight = torch.ones_like(pair_target)
        pair_weight[pair_target > 0.5] = self.positive_weight
        pair_loss = F.binary_cross_entropy_with_logits(pair_logits, pair_target, weight=pair_weight)
        division_target = (pair_target.sum(dim=1) >= 2).float()
        division_loss = F.binary_cross_entropy_with_logits(outputs["division_logits"], division_target, pos_weight=torch.tensor(self.division_weight, device=pair_logits.device))
        mitosis_loss = pair_logits.sum() * 0.0
        if outputs["mitosis_logits"].numel() > 0:
            pairs = outputs["mitosis_pairs"]
            target = []
            for p, a, b in pairs.tolist():
                mother = int(prev_ids[p].item())
                target.append(float(int(curr_parent[a].item()) == mother and int(curr_parent[b].item()) == mother))
            mitosis_target = torch.tensor(target, dtype=torch.float32, device=pair_logits.device)
            mitosis_loss = F.binary_cross_entropy_with_logits(outputs["mitosis_logits"], mitosis_target, pos_weight=torch.tensor(self.division_weight, device=pair_logits.device))
        total = pair_loss + division_loss + self.mitosis_weight * mitosis_loss
        return {"loss": total, "loss_pair": pair_loss.detach(), "loss_division": division_loss.detach(), "loss_mitosis": mitosis_loss.detach()}
