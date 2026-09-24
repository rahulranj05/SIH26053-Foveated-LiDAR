"""
F0 SPVCNN semantic segmentation model.

TorchSparse is intentionally imported lazily.  This lets the rest of the
perception package, preprocessing pipeline, tests and CPU infrastructure
remain usable when TorchSparse is unavailable.

Architecture contract
---------------------
sparse voxel features
    -> sparse stem
    -> residual sparse encoder
    -> sparse decoder
    -> voxel embedding / voxel logits
    -> point_to_voxel_inverse
    -> point embedding / point logits

The point embedding is retained explicitly for the later M13
SPVCNN + circular point-domain fusion model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .backend import (
    SparseBatch,
    SparseModelOutput,
    SparseSemanticModel,
)


NUM_CLASSES = 24
INPUT_CHANNELS = 7


def _load_torchsparse():
    try:
        import torchsparse
        import torchsparse.nn as spnn
    except Exception as exc:
        raise RuntimeError(
            "F0 SPVCNN requires MIT-HAN-Lab TorchSparse. "
            "TorchSparse is intentionally a GPU-runtime dependency."
        ) from exc

    return torchsparse, spnn


class SparseResidualBlock(nn.Module):
    """TorchSparse residual block with two 3x3 submanifold convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        indice_key: str | None = None,
    ):
        super().__init__()

        _, spnn = _load_torchsparse()

        self.conv1 = spnn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
        )
        self.bn1 = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU(True)

        self.conv2 = spnn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
        )
        self.bn2 = spnn.BatchNorm(out_channels)

        if in_channels != out_channels:
            self.proj = nn.Sequential(
                spnn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                ),
                spnn.BatchNorm(out_channels),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        identity = self.proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # TorchSparse SparseTensor supports sparse residual addition.
        out = out + identity
        out = self.relu(out)

        return out


class SparseDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        _, spnn = _load_torchsparse()

        self.down = nn.Sequential(
            spnn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            ),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )

        self.res = SparseResidualBlock(
            out_channels,
            out_channels,
        )

    def forward(self, x):
        return self.res(self.down(x))


class SparseUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        _, spnn = _load_torchsparse()

        self.up = nn.Sequential(
            spnn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                transposed=True,
            ),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
        )

        self.fuse = SparseResidualBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        torchsparse, _ = _load_torchsparse()

        x = self.up(x)

        # Sparse concatenation is coordinate-aware in TorchSparse.
        x = torchsparse.cat([x, skip])

        return self.fuse(x)


class TorchSparseSPVCNN(SparseSemanticModel):
    """
    F0 sparse point-voxel semantic network.

    The model consumes the frozen M4/M6 SparseBatch representation and
    emits both voxel-domain and point-domain features/logits.
    """

    def __init__(
        self,
        *,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        point_feature_channels: int = 64,
    ):
        super().__init__()

        if input_channels != INPUT_CHANNELS:
            raise ValueError(
                f"Frozen F0 input requires {INPUT_CHANNELS} channels; "
                f"got {input_channels}."
            )

        if num_classes != NUM_CLASSES:
            raise ValueError(
                f"Frozen ontology requires {NUM_CLASSES} logits; "
                f"got {num_classes}."
            )

        torchsparse, spnn = _load_torchsparse()

        c = base_channels

        self.stem = nn.Sequential(
            spnn.Conv3d(
                input_channels,
                c,
                kernel_size=3,
                stride=1,
            ),
            spnn.BatchNorm(c),
            spnn.ReLU(True),
            SparseResidualBlock(c, c),
        )

        self.enc1 = SparseDownBlock(c, c * 2)
        self.enc2 = SparseDownBlock(c * 2, c * 4)
        self.enc3 = SparseDownBlock(c * 4, c * 8)

        self.bottleneck = SparseResidualBlock(
            c * 8,
            c * 8,
        )

        self.dec2 = SparseUpBlock(
            c * 8,
            c * 4,
            c * 4,
        )

        self.dec1 = SparseUpBlock(
            c * 4,
            c * 2,
            c * 2,
        )

        self.dec0 = SparseUpBlock(
            c * 2,
            c,
            point_feature_channels,
        )

        self.voxel_head = nn.Linear(
            point_feature_channels,
            num_classes,
        )

        # Explicit point head retained for F0 and later replaceable by
        # the M13 point-domain hybrid fusion head.
        self.point_head = nn.Sequential(
            nn.Linear(
                point_feature_channels,
                point_feature_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                point_feature_channels,
                num_classes,
            ),
        )

        self.num_classes = num_classes
        self.point_feature_channels = point_feature_channels


    @staticmethod
    def _make_sparse_tensor(batch: SparseBatch):
        torchsparse, _ = _load_torchsparse()

        coords = batch.coordinates
        feats = batch.features

        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(
                "SPVCNN expects coordinates shaped (V,4): "
                "[batch,x,y,z]."
            )

        if feats.ndim != 2 or feats.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                "SPVCNN expects voxel features shaped (V,7)."
            )

        # TorchSparse coordinate tensors are integer-valued.
        coords = coords.to(dtype=torch.int32)

        return torchsparse.SparseTensor(
            coords=coords,
            feats=feats,
        )


    def forward(self, batch: SparseBatch) -> SparseModelOutput:
        x = self._make_sparse_tensor(batch)

        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        x = self.bottleneck(s3)

        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.dec0(x, s0)

        voxel_features = x.F

        if voxel_features.ndim != 2:
            raise RuntimeError(
                "Invalid TorchSparse voxel feature tensor."
            )

        voxel_logits = self.voxel_head(voxel_features)

        inverse = batch.point_to_voxel_inverse.long()

        if inverse.device != voxel_features.device:
            inverse = inverse.to(voxel_features.device)

        if inverse.numel() > 0:
            lo = int(inverse.min())
            hi = int(inverse.max())

            if lo < 0 or hi >= voxel_features.shape[0]:
                raise RuntimeError(
                    "point_to_voxel_inverse is outside voxel output."
                )

        # Frozen M4 identity bridge: sparse voxel -> canonical point.
        point_features = voxel_features[inverse]

        point_logits = self.point_head(point_features)

        if point_logits.shape[-1] != NUM_CLASSES:
            raise RuntimeError(
                "F0 must produce exactly 24 point logits."
            )

        return SparseModelOutput(
            voxel_features=voxel_features,
            voxel_logits=voxel_logits,
            point_features=point_features,
            point_logits=point_logits,
        )
