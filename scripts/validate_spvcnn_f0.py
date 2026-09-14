"""
Gate-2 validation for the FoveaMap F0 SPVCNN baseline.

This test validates:
    1. CUDA execution
    2. F0 model construction
    3. synthetic point-cloud forward pass
    4. cross-entropy loss
    5. backward propagation
    6. finite gradients
    7. optimizer update
    8. two-batch isolation

Validated reference environment:
    Python 3.10
    PyTorch 2.0.1+cu118
    CUDA 11.8
    TorchSparse 2.0.0b
    NVIDIA Tesla T4
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# =====================================================================
# Repository import setup
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from models.spvcnn_f0 import SPVCNNF0, count_parameters


# =====================================================================
# Configuration
# =====================================================================

SEED = 42
DEVICE = "cuda"

NUM_CLASSES = 23
IGNORE_INDEX = 0
VOXEL_SIZE = 0.05


# =====================================================================
# Synthetic point-cloud generator
# =====================================================================

def build_cloud(
    n: int,
    batch_id: int = 0,
):
    """
    Build a synthetic point cloud using the empirically validated
    TorchSparse 2.0 coordinate convention:

        [x, y, z, batch]

    Coordinates are in metres.

    Synthetic features are used only for Gate-2 systems validation.
    They are NOT the frozen feature policy for real RELLIS training.
    """

    x = (
        torch.rand(
            n,
            device=DEVICE,
        )
        * 40.0
        - 20.0
    )

    y = (
        torch.rand(
            n,
            device=DEVICE,
        )
        * 40.0
        - 20.0
    )

    z = (
        torch.rand(
            n,
            device=DEVICE,
        )
        * 4.0
        - 2.0
    )

    batch = torch.full(
        (n,),
        float(batch_id),
        device=DEVICE,
    )

    coords = torch.stack(
        [
            x,
            y,
            z,
            batch,
        ],
        dim=1,
    ).float()

    intensity = torch.rand(
        n,
        device=DEVICE,
    )

    features = torch.stack(
        [
            intensity,
            x / 20.0,
            y / 20.0,
            z / 2.0,
        ],
        dim=1,
    ).float()

    return features, coords


# =====================================================================
# Forward / backward validation
# =====================================================================

def validate_forward_backward():
    print("=" * 80)
    print("F0 FORWARD / BACKWARD")
    print("=" * 80)

    n = 12000

    features, coords = build_cloud(
        n=n,
        batch_id=0,
    )

    features.requires_grad_(True)

    labels = torch.randint(
        1,
        NUM_CLASSES,
        (n,),
        device=DEVICE,
    )

    ignore_mask = (
        torch.rand(
            n,
            device=DEVICE,
        )
        < 0.05
    )

    labels[ignore_mask] = IGNORE_INDEX

    model = SPVCNNF0(
        num_classes=NUM_CLASSES,
        voxel_size=VOXEL_SIZE,
        cr=0.5,
        in_channels=4,
        dropout=0.3,
    ).to(DEVICE)

    model.train()

    total_params, trainable_params = count_parameters(
        model
    )

    print(
        "parameters :",
        f"{total_params:,}",
    )

    print(
        "trainable  :",
        f"{trainable_params:,}",
    )

    assert total_params == 5_449_463, (
        f"Unexpected parameter count: "
        f"{total_params:,}"
    )

    result = model(
        features,
        coords,
        return_features=True,
    )

    logits = result["logits"]

    assert logits.shape == (
        n,
        NUM_CLASSES,
    ), (
        f"Unexpected logits shape: "
        f"{tuple(logits.shape)}"
    )

    assert torch.isfinite(
        logits
    ).all(), (
        "Logits contain NaN or Inf."
    )

    loss = F.cross_entropy(
        logits,
        labels,
        ignore_index=IGNORE_INDEX,
    )

    assert torch.isfinite(
        loss
    ), (
        "Loss is NaN or Inf."
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    finite_gradients = True
    gradient_tensors = 0
    nonzero_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        gradient_tensors += 1

        if not torch.isfinite(
            parameter.grad
        ).all():
            finite_gradients = False

        if (
            parameter.grad
            .abs()
            .sum()
            .item()
            > 0
        ):
            nonzero_gradients += 1

    assert finite_gradients, (
        "At least one parameter gradient "
        "contains NaN or Inf."
    )

    assert gradient_tensors > 0, (
        "No parameter gradients were produced."
    )

    assert nonzero_gradients > 0, (
        "All parameter gradients are zero."
    )

    assert features.grad is not None, (
        "Input features did not receive gradients."
    )

    assert torch.isfinite(
        features.grad
    ).all(), (
        "Input feature gradients contain NaN or Inf."
    )

    first_parameter = next(
        model.parameters()
    )

    before = (
        first_parameter
        .detach()
        .clone()
    )

    optimizer.step()

    after = (
        first_parameter
        .detach()
    )

    delta = (
        after - before
    ).abs().max()

    assert delta > 0, (
        "Optimizer did not update model parameters."
    )

    print(
        "loss       :",
        float(loss),
    )

    print(
        "grad tensors:",
        gradient_tensors,
    )

    print(
        "non-zero   :",
        nonzero_gradients,
    )

    print(
        "gradients  : PASS"
    )

    print(
        "optimizer  : PASS"
    )

    print(
        "FORWARD/BACKWARD: PASS"
    )


# =====================================================================
# Batch-isolation validation
# =====================================================================

def validate_batch_isolation():
    print()

    print("=" * 80)
    print("TWO-BATCH ISOLATION")
    print("=" * 80)

    n = 5000

    xyz = torch.empty(
        n,
        3,
        device=DEVICE,
    )

    xyz[:, 0] = (
        torch.rand(
            n,
            device=DEVICE,
        )
        * 30.0
        - 15.0
    )

    xyz[:, 1] = (
        torch.rand(
            n,
            device=DEVICE,
        )
        * 30.0
        - 15.0
    )

    xyz[:, 2] = (
        torch.rand(
            n,
            device=DEVICE,
        )
        * 4.0
        - 2.0
    )

    intensity0 = torch.rand(
        n,
        device=DEVICE,
    )

    intensity1 = (
        torch.rand(
            n,
            device=DEVICE,
        )
        * 0.25
        + 0.75
    )

    feat0 = torch.stack(
        [
            intensity0,
            xyz[:, 0] / 15.0,
            xyz[:, 1] / 15.0,
            xyz[:, 2] / 2.0,
        ],
        dim=1,
    ).float()

    feat1 = torch.stack(
        [
            intensity1,
            -xyz[:, 0] / 15.0,
            -xyz[:, 1] / 15.0,
            -xyz[:, 2] / 2.0,
        ],
        dim=1,
    ).float()

    coord0 = torch.cat(
        [
            xyz,
            torch.zeros(
                n,
                1,
                device=DEVICE,
            ),
        ],
        dim=1,
    ).float()

    coord1 = torch.cat(
        [
            xyz,
            torch.ones(
                n,
                1,
                device=DEVICE,
            ),
        ],
        dim=1,
    ).float()

    model = SPVCNNF0(
        num_classes=NUM_CLASSES,
        voxel_size=VOXEL_SIZE,
        cr=0.5,
        in_channels=4,
        dropout=0.3,
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():

        # -------------------------------------------------------------
        # Batch 0 inferred independently
        # -------------------------------------------------------------

        out0 = model(
            feat0,
            coord0,
        )

        # -------------------------------------------------------------
        # Batch 1 inferred independently.
        #
        # When inferred alone its batch index must be zero because it is
        # now the only batch in that inference call.
        # -------------------------------------------------------------

        coord1_single = (
            coord1.clone()
        )

        coord1_single[:, 3] = 0

        out1 = model(
            feat1,
            coord1_single,
        )

        # -------------------------------------------------------------
        # Joint inference:
        #
        # first cloud  -> batch 0
        # second cloud -> batch 1
        # -------------------------------------------------------------

        joint_features = torch.cat(
            [
                feat0,
                feat1,
            ],
            dim=0,
        )

        joint_coords = torch.cat(
            [
                coord0,
                coord1,
            ],
            dim=0,
        )

        joint = model(
            joint_features,
            joint_coords,
        )

    joint0 = joint[:n]
    joint1 = joint[n:]

    difference0 = (
        joint0 - out0
    ).abs()

    difference1 = (
        joint1 - out1
    ).abs()

    max_diff0 = float(
        difference0.max()
    )

    max_diff1 = float(
        difference1.max()
    )

    mean_diff0 = float(
        difference0.mean()
    )

    mean_diff1 = float(
        difference1.mean()
    )

    print(
        "batch 0 max diff :",
        max_diff0,
    )

    print(
        "batch 0 mean diff:",
        mean_diff0,
    )

    print(
        "batch 1 max diff :",
        max_diff1,
    )

    print(
        "batch 1 mean diff:",
        mean_diff1,
    )

    tolerance = 1e-4

    assert max_diff0 < tolerance, (
        "Batch 0 output changed when inferred "
        "jointly with another batch."
    )

    assert max_diff1 < tolerance, (
        "Batch 1 output changed when inferred "
        "jointly with another batch."
    )

    print(
        "TWO-BATCH ISOLATION: PASS"
    )


# =====================================================================
# Runtime validation
# =====================================================================

def validate_runtime():
    print("=" * 80)
    print("F0 GATE-2 RUNTIME")
    print("=" * 80)

    assert torch.cuda.is_available(), (
        "CUDA GPU required."
    )

    gpu_name = torch.cuda.get_device_name(
        0
    )

    capability = (
        torch.cuda.get_device_capability(
            0
        )
    )

    print(
        "PyTorch     :",
        torch.__version__,
    )

    print(
        "Torch CUDA  :",
        torch.version.cuda,
    )

    print(
        "GPU         :",
        gpu_name,
    )

    print(
        "Capability  :",
        capability,
    )

    print(
        "Coord order : [x, y, z, batch]"
    )

    print()


# =====================================================================
# Main
# =====================================================================

def main():

    validate_runtime()

    torch.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )

    validate_forward_backward()

    validate_batch_isolation()

    print()
    print("=" * 80)
    print(
        "F0 GATE-2 SYNTHETIC VALIDATION: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()