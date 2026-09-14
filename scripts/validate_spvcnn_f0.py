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

import torch
import torch.nn.functional as F

from models.spvcnn_f0 import SPVCNNF0, count_parameters


SEED = 42
DEVICE = "cuda"

NUM_CLASSES = 23
IGNORE_INDEX = 0
VOXEL_SIZE = 0.05


def build_cloud(n: int, batch_id: int = 0):
    x = torch.rand(n, device=DEVICE) * 40.0 - 20.0
    y = torch.rand(n, device=DEVICE) * 40.0 - 20.0
    z = torch.rand(n, device=DEVICE) * 4.0 - 2.0

    batch = torch.full(
        (n,),
        float(batch_id),
        device=DEVICE,
    )

    coords = torch.stack(
        [x, y, z, batch],
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


def validate_forward_backward():
    print("=" * 80)
    print("F0 FORWARD / BACKWARD")
    print("=" * 80)

    n = 12000

    features, coords = build_cloud(n)

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

    total_params, trainable_params = count_parameters(model)

    print("parameters :", f"{total_params:,}")
    print("trainable  :", f"{trainable_params:,}")

    assert total_params == 5_449_463

    result = model(
        features,
        coords,
        return_features=True,
    )

    logits = result["logits"]

    assert logits.shape == (
        n,
        NUM_CLASSES,
    )

    assert torch.isfinite(logits).all()

    loss = F.cross_entropy(
        logits,
        labels,
        ignore_index=IGNORE_INDEX,
    )

    assert torch.isfinite(loss)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    finite = True
    nonzero = 0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all():
            finite = False

        if parameter.grad.abs().sum() > 0:
            nonzero += 1

    assert finite
    assert nonzero > 0

    assert features.grad is not None
    assert torch.isfinite(
        features.grad
    ).all()

    before = (
        next(model.parameters())
        .detach()
        .clone()
    )

    optimizer.step()

    after = (
        next(model.parameters())
        .detach()
    )

    delta = (
        after - before
    ).abs().max()

    assert delta > 0

    print("loss       :", float(loss))
    print("gradients  : PASS")
    print("optimizer  : PASS")
    print("FORWARD/BACKWARD: PASS")


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
        torch.rand(n, device=DEVICE)
        * 30.0 - 15.0
    )

    xyz[:, 1] = (
        torch.rand(n, device=DEVICE)
        * 30.0 - 15.0
    )

    xyz[:, 2] = (
        torch.rand(n, device=DEVICE)
        * 4.0 - 2.0
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
        * 0.25 + 0.75
    )

    feat0 = torch.stack(
        [
            intensity0,
            xyz[:, 0] / 15.0,
            xyz[:, 1] / 15.0,
            xyz[:, 2] / 2.0,
        ],
        dim=1,
    )

    feat1 = torch.stack(
        [
            intensity1,
            -xyz[:, 0] / 15.0,
            -xyz[:, 1] / 15.0,
            -xyz[:, 2] / 2.0,
        ],
        dim=1,
    )

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
    )

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
    )

    model = SPVCNNF0(
        num_classes=NUM_CLASSES,
        voxel_size=VOXEL_SIZE,
        cr=0.5,
        in_channels=4,
        dropout=0.3,
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():

        out0 = model(
            feat0,
            coord0,
        )

        coord1_single = (
            coord1.clone()
        )

        coord1_single[:, 3] = 0

        out1 = model(
            feat1,
            coord1_single,
        )

        joint_features = torch.cat(
            [feat0, feat1],
            dim=0,
        )

        joint_coords = torch.cat(
            [coord0, coord1],
            dim=0,
        )

        joint = model(
            joint_features,
            joint_coords,
        )

    joint0 = joint[:n]
    joint1 = joint[n:]

    max_diff0 = float(
        (joint0 - out0)
        .abs()
        .max()
    )

    max_diff1 = float(
        (joint1 - out1)
        .abs()
        .max()
    )

    print(
        "batch 0 max diff:",
        max_diff0,
    )

    print(
        "batch 1 max diff:",
        max_diff1,
    )

    tolerance = 1e-4

    assert max_diff0 < tolerance
    assert max_diff1 < tolerance

    print(
        "TWO-BATCH ISOLATION: PASS"
    )


def main():

    assert torch.cuda.is_available(), (
        "CUDA GPU required."
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    validate_forward_backward()
    validate_batch_isolation()

    print()
    print("=" * 80)
    print("F0 GATE-2 SYNTHETIC VALIDATION: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()