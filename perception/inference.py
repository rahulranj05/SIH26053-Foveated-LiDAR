from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from perception.fusion.f1_hybrid import F1HybridModel
from perception.sample_pipeline import build_unified_sample
from perception.spvcnn.backend import SparseBatch


NUM_CLASSES = 24


@dataclass(frozen=True)
class PerceptionResult:
    """Point-domain output produced by F1."""

    xyz: np.ndarray
    unified_labels: np.ndarray
    confidence: np.ndarray
    model_predictions: np.ndarray
    point_logits: np.ndarray

    def validate(self) -> None:
        n = len(self.xyz)

        if self.xyz.shape != (n, 3):
            raise ValueError("xyz must have shape (N, 3)")

        if self.unified_labels.shape != (n,):
            raise ValueError("unified_labels must have shape (N,)")

        if self.confidence.shape != (n,):
            raise ValueError("confidence must have shape (N,)")

        if self.model_predictions.shape != (n,):
            raise ValueError("model_predictions must have shape (N,)")

        if self.point_logits.shape != (n, NUM_CLASSES):
            raise ValueError(
                f"point_logits must have shape (N, {NUM_CLASSES})"
            )

        if n:
            if np.any(self.model_predictions < 0) or np.any(
                self.model_predictions >= NUM_CLASSES
            ):
                raise ValueError("model predictions outside 0..23")

            if np.any(self.unified_labels < 1) or np.any(
                self.unified_labels > NUM_CLASSES
            ):
                raise ValueError("unified labels outside 1..24")

            if np.any(self.confidence < 0.0) or np.any(
                self.confidence > 1.0
            ):
                raise ValueError("confidence outside 0..1")

        if not np.isfinite(self.xyz).all():
            raise ValueError("xyz contains non-finite values")

        if not np.isfinite(self.confidence).all():
            raise ValueError("confidence contains non-finite values")

        if not np.isfinite(self.point_logits).all():
            raise ValueError("logits contain non-finite values")


class F1InferenceEngine:
    """
    Deployment wrapper around the frozen F1 hybrid model.

    Raw LiDAR XYZ + intensity is converted through the canonical
    preprocessing pipeline before being passed to SPVCNN and the
    circular neural branch.

    TorchSparse is loaded lazily by the SPVCNN implementation, so
    importing this module does not require a working TorchSparse
    runtime. Actual inference does.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"F1 checkpoint not found: {self.checkpoint_path}"
            )

        self.model: F1HybridModel | None = None

    def load(self) -> None:
        """Construct F1 and load the frozen deployment checkpoint."""

        try:
            model = F1HybridModel()
        except RuntimeError as exc:
            if "TorchSparse" in str(exc):
                raise RuntimeError(
                    "F1 inference requires the canonical TorchSparse runtime. "
                    "The recovered training environment used Linux, Python 3.10, "
                    "PyTorch 2.7.1+cu128, CUDA 12.8, and TorchSparse 2.1.0. "
                    "Preprocessing can run without TorchSparse, but F1 neural "
                    "inference cannot."
                ) from exc
            raise

        payload = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(payload, dict):
            raise ValueError("F1 checkpoint must contain a dictionary")

        state_dict = payload.get("model")

        if not isinstance(state_dict, dict):
            raise ValueError(
                "F1 checkpoint does not contain a 'model' state dictionary"
            )

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        self.model = model

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @staticmethod
    def _build_sparse_batch(sample) -> SparseBatch:
        spv = sample.spvcnn

        coordinates = np.asarray(spv.voxel_coordinates)

        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                "SPVCNN voxel coordinates must have shape (V, 3)"
            )

        # TorchSparse coordinates are [batch, x, y, z].
        # Deployment currently processes one LiDAR frame at a time,
        # therefore every voxel belongs to batch zero.
        batch_column = np.zeros(
            (coordinates.shape[0], 1),
            dtype=np.int32,
        )

        batched_coordinates = np.concatenate(
            (
                batch_column,
                coordinates.astype(np.int32, copy=False),
            ),
            axis=1,
        )

        sparse_batch = SparseBatch(
            coordinates=torch.from_numpy(batched_coordinates),
            features=torch.from_numpy(
                np.asarray(
                    spv.voxel_features,
                    dtype=np.float32,
                )
            ),
            point_to_voxel_inverse=torch.from_numpy(
                np.asarray(
                    spv.point_to_voxel_inverse,
                    dtype=np.int64,
                )
            ),
        )

        sparse_batch.validate()
        return sparse_batch

    def predict(
        self,
        *,
        xyz: np.ndarray,
        intensity: np.ndarray,
        dataset_id: str,
        source_point_id: np.ndarray | None = None,
    ) -> PerceptionResult:

        if self.model is None:
            raise RuntimeError(
                "F1 model is not loaded. Call load() first."
            )

        xyz = np.asarray(xyz, dtype=np.float32)
        intensity = np.asarray(intensity, dtype=np.float32)

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")

        if intensity.shape != (len(xyz),):
            raise ValueError("intensity must have shape (N,)")

        if source_point_id is None:
            source_point_id = np.arange(
                len(xyz),
                dtype=np.int64,
            )
        else:
            source_point_id = np.asarray(
                source_point_id,
                dtype=np.int64,
            )

        sample = build_unified_sample(
            dataset_id=dataset_id,
            split_role="inference",
            xyz=xyz,
            intensity=intensity,
            source_point_id=source_point_id,
        )

        sparse_batch = self._build_sparse_batch(sample)

        # Sparse tensors are created inside the recovered SPVCNN
        # implementation. Move ordinary tensor payloads to the selected
        # runtime device before entering the model.
        sparse_batch = SparseBatch(
            coordinates=sparse_batch.coordinates.to(self.device),
            features=sparse_batch.features.to(self.device),
            point_to_voxel_inverse=(
                sparse_batch.point_to_voxel_inverse.to(self.device)
            ),
        )

        with torch.inference_mode():
            output = self.model(
                sparse_batch,
                sample.circular,
            )

            logits = output.point_logits

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, predictions = torch.max(
                probabilities,
                dim=1,
            )

        # F1 model classes are 0..23.
        # Unified perception labels are 1..24.
        unified_labels = predictions + 1

        result = PerceptionResult(
            xyz=np.asarray(
                sample.xyz,
                dtype=np.float32,
            ).copy(),
            unified_labels=(
                unified_labels.detach().cpu().numpy().astype(
                    np.int64,
                    copy=False,
                )
            ),
            confidence=(
                confidence.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            ),
            model_predictions=(
                predictions.detach().cpu().numpy().astype(
                    np.int64,
                    copy=False,
                )
            ),
            point_logits=(
                logits.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            ),
        )

        result.validate()
        return result
