import torch
import torchsparse.nn.functional as spf

from torchsparse import SparseTensor, PointTensor


# =====================================================================
# Helpers
# =====================================================================

def _stride_tuple(stride):
    if isinstance(stride, int):
        return (stride, stride, stride)

    if len(stride) == 1:
        return (
            int(stride[0]),
            int(stride[0]),
            int(stride[0]),
        )

    return tuple(
        int(v) for v in stride[:3]
    )


def _stride_tensor(
    stride,
    device,
    dtype=torch.float32,
):
    return torch.tensor(
        _stride_tuple(stride),
        device=device,
        dtype=dtype,
    )


def _validate_point_tensor(z):
    if z.F is None:
        raise ValueError(
            "PointTensor.F is None"
        )

    if z.C is None:
        raise ValueError(
            "PointTensor.C is None"
        )

    if z.F.ndim != 2:
        raise ValueError(
            f"Expected point features [N,C], "
            f"got {tuple(z.F.shape)}"
        )

    if (
        z.C.ndim != 2
        or z.C.shape[1] != 4
    ):
        raise ValueError(
            "TorchSparse 2.0 PointTensor coordinates "
            "must be [N,4] in [x,y,z,batch] order"
        )

    if z.F.shape[0] != z.C.shape[0]:
        raise ValueError(
            "Point feature/coordinate counts differ"
        )

    if not torch.isfinite(z.F).all():
        raise ValueError(
            "Point features contain NaN/Inf"
        )

    if not torch.isfinite(z.C).all():
        raise ValueError(
            "Point coordinates contain NaN/Inf"
        )


def _ensure_caches(z):
    if z.idx_query is None:
        z.idx_query = {}

    if z.weights is None:
        z.weights = {}

    if getattr(
        z,
        "additional_features",
        None
    ) is None:
        z.additional_features = {}

    z.additional_features.setdefault(
        "idx_query",
        {}
    )

    z.additional_features.setdefault(
        "counts",
        {}
    )


# =====================================================================
# INITIAL VOXELIZATION
# =====================================================================

def initial_voxelize(
    z: PointTensor,
    voxel_size: float,
) -> SparseTensor:
    """
    TorchSparse 2.0.0b0 convention.

    Input z.C:
        [x_m, y_m, z_m, batch]

    Output SparseTensor.C:
        [vx, vy, vz, batch]

    z.C is converted in-place to continuous BASE-VOXEL coordinates:

        [x/voxel_size,
         y/voxel_size,
         z/voxel_size,
         batch]
    """

    _validate_point_tensor(z)
    _ensure_caches(z)

    if voxel_size <= 0:
        raise ValueError(
            "voxel_size must be > 0"
        )

    xyz_metric = z.C[:, :3]
    batch = z.C[:, 3:4]

    # Continuous coordinates in 5-cm voxel units.
    xyz_float = (
        xyz_metric / voxel_size
    )

    xyz_int = torch.floor(
        xyz_float
    ).int()

    # TorchSparse 2.0 order:
    # [x,y,z,batch]
    point_coords_int = torch.cat(
        [
            xyz_int,
            batch.int(),
        ],
        dim=1,
    )

    point_hash = spf.sphash(
        point_coords_int
    )

    sparse_hash = torch.unique(
        point_hash
    )

    idx_query = spf.sphashquery(
        point_hash,
        sparse_hash,
    )

    if bool(
        (idx_query < 0).any()
    ):
        raise RuntimeError(
            "initial_voxelize generated "
            "invalid voxel indices"
        )

    counts = spf.spcount(
        idx_query.int(),
        sparse_hash.shape[0],
    )

    inserted_coords = spf.spvoxelize(
        point_coords_int.float(),
        idx_query,
        counts,
    )

    inserted_coords = torch.round(
        inserted_coords
    ).int()

    inserted_feats = spf.spvoxelize(
        z.F,
        idx_query,
        counts,
    )

    out = SparseTensor(
        feats=inserted_feats,
        coords=inserted_coords,
        stride=1,
    )

    # Keep continuous base-grid coordinates on PointTensor.
    z.C = torch.cat(
        [
            xyz_float,
            batch.float(),
        ],
        dim=1,
    )

    key = _stride_tuple(
        out.s
    )

    z.additional_features[
        "idx_query"
    ][key] = idx_query

    z.additional_features[
        "counts"
    ][key] = counts

    return out


# =====================================================================
# POINT -> VOXEL
# =====================================================================

def point_to_voxel(
    x: SparseTensor,
    z: PointTensor,
) -> SparseTensor:
    """
    Robust PointTensor -> SparseTensor projection for the installed
    TorchSparse 2.0.0b build.

    Coordinate convention:
        [x, y, z, batch]

    Sparse coordinates retain BASE-LATTICE scale after downsampling.

    Important:
        After sparse convolutions, floor(point / stride) * stride is
        not guaranteed to exist in x.C for every original point.

    Therefore we:
        1. construct the same 8 lattice neighbours used by
           voxel_to_point(),
        2. query the ACTUAL active sparse coordinate set x.C,
        3. use the floor neighbour when present,
        4. otherwise select the geometrically closest valid neighbour,
        5. scatter-mean point features into those sparse voxels.

    This eliminates assumptions about sparse-convolution active-set
    generation while preserving hard point->voxel assignment.
    """

    _validate_point_tensor(z)
    _ensure_caches(z)

    stride = _stride_tensor(
        x.s,
        z.C.device,
        dtype=z.C.dtype,
    )

    # --------------------------------------------------------------
    # Point locations in the current stride coordinate system
    # --------------------------------------------------------------

    scaled_xyz = (
        z.C[:, :3]
        / stride
    )

    base = torch.floor(
        scaled_xyz
    ).int()

    frac = (
        scaled_xyz
        - base.float()
    )

    offsets = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.int32,
        device=z.C.device,
    )

    # --------------------------------------------------------------
    # Old TorchSparse/base-lattice semantics:
    #
    # (floor(point / stride) + offset) * stride
    # --------------------------------------------------------------

    neighbour_xyz = (
        base[:, None, :]
        +
        offsets[None, :, :]
    )

    neighbour_xyz = (
        neighbour_xyz
        *
        stride.int()[None, None, :]
    )

    batch = (
        z.C[:, 3]
        .int()
    )

    neighbour_batch = (
        batch[:, None, None]
        .expand(-1, 8, 1)
    )

    query_coords = torch.cat(
        [
            neighbour_xyz,
            neighbour_batch,
        ],
        dim=2,
    ).reshape(-1, 4).contiguous()

    # --------------------------------------------------------------
    # Hash against the ACTUAL sparse active set
    # --------------------------------------------------------------

    query_hash = spf.sphash(
        query_coords
    )

    sparse_hash = spf.sphash(
        x.C.int()
    )

    idx = spf.sphashquery(
        query_hash,
        sparse_hash,
    ).reshape(-1, 8)

    valid = (
        idx >= 0
    )

    has_any = valid.any(
        dim=1
    )

    if not bool(has_any.all()):
        missing = int(
            (~has_any)
            .sum()
            .item()
        )

        raise RuntimeError(
            f"point_to_voxel: {missing}/{z.C.shape[0]} "
            f"points have no active sparse neighbour "
            f"at stride {_stride_tuple(x.s)}"
        )

    # --------------------------------------------------------------
    # Geometric trilinear scores
    #
    # Larger score = closer lattice corner.
    # --------------------------------------------------------------

    diff = torch.abs(
        frac[:, None, :]
        -
        offsets.float()[None, :, :]
    )

    weights = torch.prod(
        1.0 - diff,
        dim=2,
    )

    weights = weights.masked_fill(
        ~valid,
        -1.0,
    )

    # --------------------------------------------------------------
    # Prefer the normal "floor voxel" whenever it exists.
    #
    # Offset index 0 = [0,0,0].
    #
    # Only use nearest valid neighbour as fallback when that exact
    # sparse coordinate is absent.
    # --------------------------------------------------------------

    floor_valid = valid[:, 0]

    best_fallback = weights.argmax(
        dim=1
    )

    chosen_corner = torch.where(
        floor_valid,
        torch.zeros_like(best_fallback),
        best_fallback,
    )

    rows = torch.arange(
        z.C.shape[0],
        device=z.C.device,
    )

    chosen_idx = idx[
        rows,
        chosen_corner
    ].long()

    if bool(
        (chosen_idx < 0).any()
    ):
        raise RuntimeError(
            "Internal point_to_voxel error: "
            "selected an invalid sparse index"
        )

    # --------------------------------------------------------------
    # Scatter-mean
    #
    # No torch_scatter dependency required.
    # Differentiable w.r.t. z.F.
    # --------------------------------------------------------------

    n_voxels = x.C.shape[0]
    channels = z.F.shape[1]

    sparse_feat = torch.zeros(
        n_voxels,
        channels,
        dtype=z.F.dtype,
        device=z.F.device,
    )

    sparse_feat.index_add_(
        0,
        chosen_idx,
        z.F,
    )

    counts = torch.zeros(
        n_voxels,
        dtype=z.F.dtype,
        device=z.F.device,
    )

    counts.index_add_(
        0,
        chosen_idx,
        torch.ones(
            chosen_idx.shape[0],
            dtype=z.F.dtype,
            device=z.F.device,
        ),
    )

    occupied = (
        counts > 0
    )

    sparse_feat[occupied] = (
        sparse_feat[occupied]
        /
        counts[occupied, None]
    )

    # Sparse voxels that receive no original points simply remain zero.
    # This is legitimate: x.C is the convolution active set, which may
    # contain sites generated by sparse convolution rather than direct
    # input-point occupancy.

    out = SparseTensor(
        feats=sparse_feat,
        coords=x.C,
        stride=x.s,
    )

    if hasattr(x, "cmaps"):
        out.cmaps = x.cmaps

    if hasattr(x, "kmaps"):
        out.kmaps = x.kmaps

    return out


# =====================================================================
# VOXEL -> POINT
# =====================================================================

def voxel_to_point(
    x: SparseTensor,
    z: PointTensor,
    nearest=False,
) -> PointTensor:
    """
    Trilinear sparse voxel -> point interpolation.

    TorchSparse 2.0 coordinate system:

        SparseTensor.C = [x,y,z,batch]

    At stride S, occupied lattice points are spaced S base-grid units.

    For each point:

        scaled = xyz / S
        base   = floor(scaled)

    Candidate sparse coordinates:

        (base + {0,1}^3) * S
    """

    _validate_point_tensor(z)
    _ensure_caches(z)

    key = _stride_tuple(
        x.s
    )

    cached_idx = z.idx_query.get(
        key
    )

    cached_weights = z.weights.get(
        key
    )

    if (
        cached_idx is None
        or cached_weights is None
    ):
        stride = _stride_tensor(
            x.s,
            z.C.device,
            dtype=z.C.dtype,
        )

        # Coordinates in CURRENT stride units.
        scaled_xyz = (
            z.C[:, :3]
            / stride
        )

        base = torch.floor(
            scaled_xyz
        ).int()

        offsets = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.int32,
            device=z.C.device,
        )

        # Convert candidate corners back to BASE lattice.
        neighbour_xyz = (
            base[:, None, :]
            +
            offsets[None, :, :]
        )

        neighbour_xyz = (
            neighbour_xyz
            *
            stride.int()[
                None,
                None,
                :
            ]
        )

        batch = (
            z.C[:, 3]
            .int()
        )

        neighbour_batch = (
            batch[
                :,
                None,
                None
            ]
            .expand(
                -1,
                8,
                1
            )
        )

        query_coords = torch.cat(
            [
                neighbour_xyz,
                neighbour_batch,
            ],
            dim=2,
        )

        query_coords = (
            query_coords
            .reshape(-1, 4)
            .contiguous()
        )

        query_hash = spf.sphash(
            query_coords
        )

        sparse_hash = spf.sphash(
            x.C.int()
        )

        idx = spf.sphashquery(
            query_hash,
            sparse_hash,
        )

        idx = idx.reshape(
            -1,
            8
        )

        # -------------------------------------------------------------
        # Trilinear weights
        # -------------------------------------------------------------

        frac = (
            scaled_xyz
            -
            base.float()
        )

        offsets_float = (
            offsets.float()
        )

        diff = torch.abs(
            frac[:, None, :]
            -
            offsets_float[
                None,
                :,
                :
            ]
        )

        weights = torch.prod(
            1.0 - diff,
            dim=2,
        )

        valid = (
            idx >= 0
        )

        weights = (
            weights.masked_fill(
                ~valid,
                0.0
            )
        )

        if nearest:
            candidate = (
                weights.clone()
            )

            candidate[
                ~valid
            ] = -1.0

            best = candidate.argmax(
                dim=1
            )

            mask = torch.zeros_like(
                weights
            )

            rows = torch.arange(
                weights.shape[0],
                device=weights.device,
            )

            mask[
                rows,
                best
            ] = 1.0

            weights = (
                weights
                *
                mask
            )

        weight_sum = weights.sum(
            dim=1,
            keepdim=True,
        )

        has_neighbour = (
            weight_sum[:, 0]
            > 0
        )

        if not bool(
            has_neighbour.all()
        ):
            missing = int(
                (~has_neighbour)
                .sum()
                .item()
            )

            raise RuntimeError(
                f"voxel_to_point failed: "
                f"{missing}/{z.C.shape[0]} "
                f"points have no sparse "
                f"neighbour at stride {key}"
            )

        weights = (
            weights
            /
            weight_sum.clamp_min(
                1e-12
            )
        )

        cached_idx = (
            idx.contiguous()
        )

        cached_weights = (
            weights
            .contiguous()
            .float()
        )

        z.idx_query[
            key
        ] = cached_idx

        z.weights[
            key
        ] = cached_weights

    # -------------------------------------------------------------
    # Explicit differentiable sparse interpolation.
    #
    # Matrix:
    #
    #       [N_points x N_voxels]
    #
    # output:
    #
    #       M @ x.F
    #
    # This avoids version-specific spdevoxelize layouts.
    # -------------------------------------------------------------

    n_points = (
        z.C.shape[0]
    )

    n_voxels = (
        x.F.shape[0]
    )

    valid = (
        cached_idx >= 0
    )

    point_ids = (
        torch.arange(
            n_points,
            device=x.F.device,
            dtype=torch.long,
        )[
            :,
            None
        ]
        .expand(
            -1,
            8
        )
    )

    row = point_ids[
        valid
    ]

    col = (
        cached_idx[
            valid
        ]
        .long()
    )

    val = (
        cached_weights[
            valid
        ]
        .to(
            dtype=x.F.dtype
        )
    )

    sparse_indices = torch.stack(
        [
            row,
            col,
        ],
        dim=0,
    )

    interpolation = (
        torch.sparse_coo_tensor(
            sparse_indices,
            val,
            size=(
                n_points,
                n_voxels,
            ),
            dtype=x.F.dtype,
            device=x.F.device,
        )
        .coalesce()
    )

    new_feat = torch.sparse.mm(
        interpolation,
        x.F,
    )

    if new_feat.shape != (
        n_points,
        x.F.shape[1],
    ):
        raise RuntimeError(
            "Unexpected voxel_to_point "
            f"shape: {tuple(new_feat.shape)}"
        )

    if not torch.isfinite(
        new_feat
    ).all():
        raise RuntimeError(
            "voxel_to_point generated NaN/Inf"
        )

    out = PointTensor(
        feats=new_feat,
        coords=z.C,
        idx_query=z.idx_query,
        weights=z.weights,
    )

    out.additional_features = (
        z.additional_features
    )

    return out
