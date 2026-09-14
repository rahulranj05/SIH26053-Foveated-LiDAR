import torch
import torch.nn as nn

import torchsparse
import torchsparse.nn as spnn

from torchsparse import PointTensor

from models.spv_adapter import (
    initial_voxelize,
    point_to_voxel,
    voxel_to_point,
)


# =====================================================================
# Sparse building blocks
# =====================================================================

class BasicConvolutionBlock(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        ks=3,
        stride=1,
        dilation=1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        ks=3,
        stride=1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                stride=stride,
                transposed=True,
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        ks=3,
        stride=1,
        dilation=1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            spnn.Conv3d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),

            spnn.Conv3d(
                outc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()

        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc,
                    outc,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.net(x)
        identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# =====================================================================
# F0 â€” SPVCNN-30GMAC-class baseline
# =====================================================================

class SPVCNNF0(nn.Module):
    """
    FoveaMap F0 baseline.

    Input:
        feats:
            [N, 4]
            recommended:
                intensity,
                x_rel,
                y_rel,
                z_rel

            For the first compatibility test arbitrary 4-D features are okay.

        coords:
            [N, 4]
            PointTensor convention:
                [x, y, z, batch]

            xyz are metric LiDAR coordinates.

    Output:
        logits:
            [N, num_classes]

    Default:
        num_classes = 23
        voxel_size  = 0.05 m
        cr          = 0.5
    """

    def __init__(
        self,
        num_classes=23,
        voxel_size=0.05,
        cr=0.5,
        in_channels=4,
        dropout=0.3,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.voxel_size = float(voxel_size)
        self.cr = float(cr)
        self.in_channels = int(in_channels)

        if self.num_classes <= 1:
            raise ValueError(
                "num_classes must be > 1"
            )

        if self.voxel_size <= 0:
            raise ValueError(
                "voxel_size must be > 0"
            )

        # Official manually designed SPVCNN channel schedule.
        base_channels = [
            32,
            32,
            64,
            128,
            256,
            256,
            128,
            96,
            96,
        ]

        cs = [
            int(self.cr * c)
            for c in base_channels
        ]

        if min(cs) <= 0:
            raise ValueError(
                f"Invalid channel ratio cr={self.cr}"
            )

        self.channels = cs

        # -------------------------------------------------------------
        # Stem
        # -------------------------------------------------------------

        self.stem = nn.Sequential(
            spnn.Conv3d(
                self.in_channels,
                cs[0],
                kernel_size=3,
                stride=1,
            ),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),

            spnn.Conv3d(
                cs[0],
                cs[0],
                kernel_size=3,
                stride=1,
            ),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        # -------------------------------------------------------------
        # Encoder
        # -------------------------------------------------------------

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(
                cs[0],
                cs[0],
                ks=2,
                stride=2,
            ),

            ResidualBlock(
                cs[0],
                cs[1],
            ),

            ResidualBlock(
                cs[1],
                cs[1],
            ),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(
                cs[1],
                cs[1],
                ks=2,
                stride=2,
            ),

            ResidualBlock(
                cs[1],
                cs[2],
            ),

            ResidualBlock(
                cs[2],
                cs[2],
            ),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(
                cs[2],
                cs[2],
                ks=2,
                stride=2,
            ),

            ResidualBlock(
                cs[2],
                cs[3],
            ),

            ResidualBlock(
                cs[3],
                cs[3],
            ),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(
                cs[3],
                cs[3],
                ks=2,
                stride=2,
            ),

            ResidualBlock(
                cs[3],
                cs[4],
            ),

            ResidualBlock(
                cs[4],
                cs[4],
            ),
        )

        # -------------------------------------------------------------
        # Decoder
        # -------------------------------------------------------------

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(
                cs[4],
                cs[5],
                ks=2,
                stride=2,
            ),

            nn.Sequential(
                ResidualBlock(
                    cs[5] + cs[3],
                    cs[5],
                ),

                ResidualBlock(
                    cs[5],
                    cs[5],
                ),
            ),
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(
                cs[5],
                cs[6],
                ks=2,
                stride=2,
            ),

            nn.Sequential(
                ResidualBlock(
                    cs[6] + cs[2],
                    cs[6],
                ),

                ResidualBlock(
                    cs[6],
                    cs[6],
                ),
            ),
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(
                cs[6],
                cs[7],
                ks=2,
                stride=2,
            ),

            nn.Sequential(
                ResidualBlock(
                    cs[7] + cs[1],
                    cs[7],
                ),

                ResidualBlock(
                    cs[7],
                    cs[7],
                ),
            ),
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(
                cs[7],
                cs[8],
                ks=2,
                stride=2,
            ),

            nn.Sequential(
                ResidualBlock(
                    cs[8] + cs[0],
                    cs[8],
                ),

                ResidualBlock(
                    cs[8],
                    cs[8],
                ),
            ),
        ])

        # -------------------------------------------------------------
        # Point branch
        # -------------------------------------------------------------

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    cs[0],
                    cs[4],
                ),
                nn.BatchNorm1d(
                    cs[4]
                ),
                nn.ReLU(True),
            ),

            nn.Sequential(
                nn.Linear(
                    cs[4],
                    cs[6],
                ),
                nn.BatchNorm1d(
                    cs[6]
                ),
                nn.ReLU(True),
            ),

            nn.Sequential(
                nn.Linear(
                    cs[6],
                    cs[8],
                ),
                nn.BatchNorm1d(
                    cs[8]
                ),
                nn.ReLU(True),
            ),
        ])

        self.dropout = nn.Dropout(
            p=float(dropout),
            inplace=False,
        )

        self.classifier = nn.Linear(
            cs[8],
            self.num_classes,
        )

        self._initialize_weights()

    # =================================================================
    # Initialization
    # =================================================================

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(
                m,
                nn.BatchNorm1d
            ):
                if m.weight is not None:
                    nn.init.constant_(
                        m.weight,
                        1.0
                    )

                if m.bias is not None:
                    nn.init.constant_(
                        m.bias,
                        0.0
                    )

    # =================================================================
    # Input validation
    # =================================================================

    def _validate_input(
        self,
        feats,
        coords,
    ):

        if feats.ndim != 2:
            raise ValueError(
                "feats must be [N,C]"
            )

        if coords.ndim != 2:
            raise ValueError(
                "coords must be [N,4]"
            )

        if coords.shape[1] != 4:
            raise ValueError(
                "coords must use "
                "[x,y,z,batch]"
            )

        if feats.shape[0] != coords.shape[0]:
            raise ValueError(
                "Feature/coordinate "
                "counts differ"
            )

        if feats.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected "
                f"{self.in_channels} "
                f"features, got "
                f"{feats.shape[1]}"
            )

        if feats.shape[0] == 0:
            raise ValueError(
                "Empty point cloud"
            )

        if not torch.isfinite(
            feats
        ).all():
            raise ValueError(
                "NaN/Inf in features"
            )

        if not torch.isfinite(
            coords
        ).all():
            raise ValueError(
                "NaN/Inf in coordinates"
            )

        # Batch IDs must represent integers.
        b = coords[:, 3]

        if not torch.allclose(
            b,
            torch.round(b),
        ):
            raise ValueError(
                "Batch IDs must be integer-valued"
            )

    # =================================================================
    # Forward
    # =================================================================

    def forward(
        self,
        feats,
        coords,
        return_features=False,
    ):

        self._validate_input(
            feats,
            coords,
        )

        # -------------------------------------------------------------
        # Point representation
        # -------------------------------------------------------------

        z = PointTensor(
            feats=feats,
            coords=coords.float(),
        )

        # -------------------------------------------------------------
        # Initial sparse voxelization
        # -------------------------------------------------------------

        x0 = initial_voxelize(
            z,
            voxel_size=self.voxel_size,
        )

        # Sparse stem
        x0 = self.stem(x0)

        # First fine-resolution point representation
        z0 = voxel_to_point(
            x0,
            z,
            nearest=False,
        )

        # -------------------------------------------------------------
        # Encoder
        # -------------------------------------------------------------

        x1 = point_to_voxel(
            x0,
            z0,
        )

        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # -------------------------------------------------------------
        # Point fusion #1
        #
        # deep voxel features
        #       +
        # transformed high-resolution point features
        # -------------------------------------------------------------

        z1 = voxel_to_point(
            x4,
            z0,
            nearest=False,
        )

        z1.F = (
            z1.F
            +
            self.point_transforms[0](
                z0.F
            )
        )

        # -------------------------------------------------------------
        # Decoder stage 1
        # -------------------------------------------------------------

        y1 = point_to_voxel(
            x4,
            z1,
        )

        y1.F = self.dropout(
            y1.F
        )

        y1 = self.up1[0](y1)

        y1 = torchsparse.cat([
            y1,
            x3,
        ])

        y1 = self.up1[1](y1)

        # -------------------------------------------------------------
        # Decoder stage 2
        # -------------------------------------------------------------

        y2 = self.up2[0](y1)

        y2 = torchsparse.cat([
            y2,
            x2,
        ])

        y2 = self.up2[1](y2)

        # -------------------------------------------------------------
        # Point fusion #2
        # -------------------------------------------------------------

        z2 = voxel_to_point(
            y2,
            z1,
            nearest=False,
        )

        z2.F = (
            z2.F
            +
            self.point_transforms[1](
                z1.F
            )
        )

        # -------------------------------------------------------------
        # Decoder stage 3
        # -------------------------------------------------------------

        y3 = point_to_voxel(
            y2,
            z2,
        )

        y3.F = self.dropout(
            y3.F
        )

        y3 = self.up3[0](y3)

        y3 = torchsparse.cat([
            y3,
            x1,
        ])

        y3 = self.up3[1](y3)

        # -------------------------------------------------------------
        # Decoder stage 4
        # -------------------------------------------------------------

        y4 = self.up4[0](y3)

        y4 = torchsparse.cat([
            y4,
            x0,
        ])

        y4 = self.up4[1](y4)

        # -------------------------------------------------------------
        # Point fusion #3
        # -------------------------------------------------------------

        z3 = voxel_to_point(
            y4,
            z2,
            nearest=False,
        )

        z3.F = (
            z3.F
            +
            self.point_transforms[2](
                z2.F
            )
        )

        # -------------------------------------------------------------
        # Semantic segmentation head
        # -------------------------------------------------------------

        logits = self.classifier(
            z3.F
        )

        if logits.shape[0] != feats.shape[0]:
            raise RuntimeError(
                "SPVCNN changed point count: "
                f"{feats.shape[0]} â†’ "
                f"{logits.shape[0]}"
            )

        if logits.shape[1] != self.num_classes:
            raise RuntimeError(
                "Classifier output dimension "
                "is incorrect"
            )

        if not torch.isfinite(
            logits
        ).all():
            raise RuntimeError(
                "SPVCNN produced NaN/Inf"
            )

        if return_features:
            return {
                "logits": logits,
                "point_features": z3.F,
                "point_coords": z3.C,
            }

        return logits


# =====================================================================
# Utilities
# =====================================================================

def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable

