from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from perception.circular.input_builder import CircularInput
from perception.spvcnn.backend import (
    SparseBatch,
    SparseModelOutput,
)
from perception.spvcnn.torchsparse_spvcnn import (
    TorchSparseSPVCNN,
)


NUM_CLASSES = 24
INPUT_CHANNELS = 7
POINT_FEATURE_CHANNELS = 64
CIRCULAR_FEATURE_CHANNELS = 64
FUSION_FEATURE_CHANNELS = 64
PROJECTION_WIDTH = 2048


class CircularConvBlock(nn.Module):
    """
    Small circular/range-view feature extractor.

    Horizontal axis uses explicit circular padding so the azimuth seam
    remains continuous. Vertical padding is ordinary zero padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0),
        )
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def circular_pad(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(x.shape)}"
            )

        if x.shape[-1] <= pad:
            raise ValueError(
                "Circular width is too small for requested padding"
            )

        left = x[..., -pad:]
        right = x[..., :pad]
        return torch.cat((left, x, right), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.circular_pad(x, pad=1)
        return self.activation(self.conv(x))


class CircularNeuralBranch(nn.Module):
    """
    Fresh neural branch operating on the existing circular projection.

    Input:
        [B, 7, H, 2048]

    Output:
        [B, 64, H, 2048]

    H is dataset-dependent and remains either 32 or 64.
    """

    def __init__(
        self,
        *,
        input_channels: int = INPUT_CHANNELS,
        output_channels: int = CIRCULAR_FEATURE_CHANNELS,
    ) -> None:
        super().__init__()

        if input_channels != INPUT_CHANNELS:
            raise ValueError(
                f"F1 expects {INPUT_CHANNELS} input channels"
            )

        self.block1 = CircularConvBlock(
            input_channels,
            32,
        )
        self.block2 = CircularConvBlock(
            32,
            64,
        )
        self.block3 = CircularConvBlock(
            64,
            output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(x.shape)}"
            )

        if x.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} channels, got {x.shape[1]}"
            )

        if x.shape[-1] != PROJECTION_WIDTH:
            raise ValueError(
                f"Expected width {PROJECTION_WIDTH}, got {x.shape[-1]}"
            )

        if x.shape[-2] not in (32, 64):
            raise ValueError(
                f"Expected circular height 32 or 64, got {x.shape[-2]}"
            )

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


def lift_circular_features_to_points(
    pixel_features: torch.Tensor,
    circular_input: CircularInput,
) -> torch.Tensor:
    """
    Differentiable Torch equivalent of the existing NumPy
    point-domain circular lift.

    Only z-buffer winners receive pixel features.
    Losers and out-of-FOV points receive zeros.

    IMPORTANT:
    This operates using the already-frozen projection mappings.
    It does not recompute projection geometry.
    """

    if pixel_features.ndim == 3:
        expected_hwc = pixel_features
    elif pixel_features.ndim == 4 and pixel_features.shape[0] == 1:
        expected_hwc = pixel_features[0].permute(1, 2, 0)
    else:
        raise ValueError(
            "pixel_features must be [H,W,C] or [1,C,H,W]"
        )

    h, w, channels = expected_hwc.shape

    if h != circular_input.image_features.shape[0]:
        raise ValueError("Circular feature height mismatch")

    if w != circular_input.image_features.shape[1]:
        raise ValueError("Circular feature width mismatch")

    point_to_pixel = torch.as_tensor(
        circular_input.point_to_pixel,
        dtype=torch.long,
        device=expected_hwc.device,
    )

    winner_mask = torch.as_tensor(
        circular_input.winner_mask,
        dtype=torch.bool,
        device=expected_hwc.device,
    )

    num_points = int(winner_mask.shape[0])

    if point_to_pixel.shape != (num_points, 2):
        raise ValueError(
            "point_to_pixel shape does not match winner_mask"
        )

    winner_points = torch.nonzero(
        winner_mask,
        as_tuple=False,
    ).flatten()

    lifted = torch.zeros(
        (num_points, channels),
        dtype=expected_hwc.dtype,
        device=expected_hwc.device,
    )

    if winner_points.numel() == 0:
        return lifted

    rows = point_to_pixel[winner_points, 0]
    cols = point_to_pixel[winner_points, 1]

    if torch.any(rows < 0) or torch.any(rows >= h):
        raise ValueError("Winner row index outside circular image")

    if torch.any(cols < 0) or torch.any(cols >= w):
        raise ValueError("Winner column index outside circular image")

    winner_features = expected_hwc[rows, cols]

    return lifted.index_copy(
        0,
        winner_points,
        winner_features,
    )


class F1HybridModel(nn.Module):
    """
    F1-A hybrid semantic model.

    Same canonical LiDAR frame is represented in two ways:

        canonical points
             /        \
            /          \
      SPVCNN branch   circular branch
            \          /
             \        /
          point-domain fusion
                 |
              24 logits
    """

    def __init__(
        self,
        *,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        point_feature_channels: int = POINT_FEATURE_CHANNELS,
        circular_feature_channels: int = CIRCULAR_FEATURE_CHANNELS,
        fusion_feature_channels: int = FUSION_FEATURE_CHANNELS,
    ) -> None:
        super().__init__()

        if input_channels != INPUT_CHANNELS:
            raise ValueError(
                f"F1 expects {INPUT_CHANNELS} input channels"
            )

        if num_classes != NUM_CLASSES:
            raise ValueError(
                f"F1 expects {NUM_CLASSES} classes"
            )

        if point_feature_channels != POINT_FEATURE_CHANNELS:
            raise ValueError(
                f"F1 expects SPVCNN point embedding "
                f"width {POINT_FEATURE_CHANNELS}"
            )

        if circular_feature_channels != CIRCULAR_FEATURE_CHANNELS:
            raise ValueError(
                f"F1 expects circular embedding "
                f"width {CIRCULAR_FEATURE_CHANNELS}"
            )

        if fusion_feature_channels != FUSION_FEATURE_CHANNELS:
            raise ValueError(
                f"F1 expects fusion width {FUSION_FEATURE_CHANNELS}"
            )

        self.spvcnn = TorchSparseSPVCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            point_feature_channels=point_feature_channels,
        )

        self.circular = CircularNeuralBranch(
            input_channels=input_channels,
            output_channels=circular_feature_channels,
        )

        fusion_input_channels = (
            point_feature_channels
            + circular_feature_channels
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_input_channels,
                128,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                128,
                fusion_feature_channels,
            ),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(
            fusion_feature_channels,
            num_classes,
        )

    @staticmethod
    def _circular_input_to_torch(
        circular_input: CircularInput,
        device: torch.device,
    ) -> torch.Tensor:
        image = torch.as_tensor(
            circular_input.image_features,
            dtype=torch.float32,
            device=device,
        )

        if image.ndim != 3:
            raise ValueError(
                f"Expected circular image [H,W,C], got {tuple(image.shape)}"
            )

        if image.shape[2] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} circular channels, "
                f"got {image.shape[2]}"
            )

        return image.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        sparse_batch: SparseBatch,
        circular_input: CircularInput,
    ) -> SparseModelOutput:

        spv_output = self.spvcnn(
            sparse_batch
        )

        image = self._circular_input_to_torch(
            circular_input,
            spv_output.point_features.device,
        )

        circular_pixel_features = self.circular(
            image
        )

        circular_point_features = (
            lift_circular_features_to_points(
                circular_pixel_features,
                circular_input,
            )
        )

        if (
            circular_point_features.shape[0]
            != spv_output.point_features.shape[0]
        ):
            raise ValueError(
                "SPVCNN and circular branches have different "
                "point counts"
            )

        fused_point_features = self.fusion(
            torch.cat(
                (
                    spv_output.point_features,
                    circular_point_features,
                ),
                dim=1,
            )
        )

        point_logits = self.classifier(
            fused_point_features
        )

        return SparseModelOutput(
            voxel_features=spv_output.voxel_features,
            voxel_logits=spv_output.voxel_logits,
            point_features=fused_point_features,
            point_logits=point_logits,
        )


def load_f0_spvcnn_weights(
    model: F1HybridModel,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    """
    Initialize only the SPVCNN branch from the canonical F0 checkpoint.

    F0 optimizer/sampler/RNG state is deliberately not restored.
    """

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"F0 checkpoint not found: {path}"
        )

    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if "model" not in payload:
        raise KeyError(
            "F0 checkpoint does not contain a 'model' state"
        )

    model.spvcnn.load_state_dict(
        payload["model"],
        strict=True,
    )

    return payload

