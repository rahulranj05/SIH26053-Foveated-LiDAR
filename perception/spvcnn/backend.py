from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SparseBatch:
    coordinates: torch.Tensor
    features: torch.Tensor
    point_to_voxel_inverse: torch.Tensor

    def validate(self) -> None:
        if self.coordinates.ndim != 2:
            raise ValueError(
                "coordinates must have shape (V, D)"
            )

        if self.features.ndim != 2:
            raise ValueError(
                "features must have shape (V, C)"
            )

        if len(self.coordinates) != len(self.features):
            raise ValueError(
                "coordinate/feature voxel count mismatch"
            )

        if self.point_to_voxel_inverse.ndim != 1:
            raise ValueError(
                "point_to_voxel_inverse must have shape (N,)"
            )

        if self.coordinates.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "sparse coordinates must be integer"
            )

        if not torch.isfinite(self.features).all():
            raise ValueError(
                "sparse features must be finite"
            )

        if len(self.features) > 0:
            inv = self.point_to_voxel_inverse

            if torch.any(inv < 0):
                raise ValueError(
                    "negative point-to-voxel index"
                )

            if torch.any(inv >= len(self.features)):
                raise ValueError(
                    "point-to-voxel index out of bounds"
                )


@dataclass
class SparseModelOutput:
    voxel_features: torch.Tensor
    voxel_logits: torch.Tensor
    point_features: torch.Tensor
    point_logits: torch.Tensor

    def validate(
        self,
        *,
        num_points: int,
        num_voxels: int,
        num_classes: int,
    ) -> None:

        if self.voxel_features.ndim != 2:
            raise ValueError(
                "voxel_features must be 2-D"
            )

        if self.point_features.ndim != 2:
            raise ValueError(
                "point_features must be 2-D"
            )

        if self.voxel_logits.shape != (
            num_voxels,
            num_classes,
        ):
            raise ValueError(
                "invalid voxel logit shape"
            )

        if self.point_logits.shape != (
            num_points,
            num_classes,
        ):
            raise ValueError(
                "invalid point logit shape"
            )

        if len(self.voxel_features) != num_voxels:
            raise ValueError(
                "voxel feature count mismatch"
            )

        if len(self.point_features) != num_points:
            raise ValueError(
                "point feature count mismatch"
            )

        for tensor in (
            self.voxel_features,
            self.voxel_logits,
            self.point_features,
            self.point_logits,
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    "non-finite model output"
                )


class SparseSemanticModel(nn.Module, ABC):

    num_classes: int

    @abstractmethod
    def forward(
        self,
        batch: SparseBatch,
    ) -> SparseModelOutput:
        raise NotImplementedError
