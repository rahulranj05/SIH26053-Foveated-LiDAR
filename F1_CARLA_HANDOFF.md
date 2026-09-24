# SIH26053 — F1 / CARLA Integration Handoff

## Status

Branch:

    advit-mapping

Integration regression status:

    832 tests passing

The repository now contains the integration boundary required to connect
CARLA LiDAR, F1 semantic perception, FoveaMap MappingBackend, and the
existing backend/frontend interface.

---

## Pipeline

The intended runtime flow is:

    CARLA LiDAR
        |
        | raw float32 [x, y, z, intensity]
        v
    perception/carla_pipeline.py
        |
        +--> preserve XYZ + intensity
        |
        v
    perception/inference.py
        |
        v
    F1HybridModel
        |
        +--> SPVCNN branch
        +--> Circular / Range-View Neural Branch
        +--> point-domain fusion
        |
        v
    24-class unified prediction
        |
        | semantic adapter
        v
    FoveaMap semantic classes 0..5
        |
        v
    SensorFrame
        |
        +--> semantic_labels
        +--> semantic_confidence
        +--> vehicle_state
        +--> timestamp
        +--> frame_id
        |
        v
    MappingBackend
        |
        v
    BackendOutput schema 1.0
        |
        v
    WebSocket / frontend

---

## Important Files

### Perception

    perception/inference.py

Deployment wrapper for F1.

Responsibilities:

- construct F1HybridModel
- load checkpoint
- build SparseBatch
- execute inference
- softmax point logits
- produce confidence
- convert model predictions 0..23 to unified IDs 1..24

### CARLA Integration

    perception/carla_pipeline.py

Integration boundary between CARLA, F1, SensorFrame and MappingBackend.

It preserves CARLA intensity without modifying the existing XYZ-only
CARLA mapping adapter.

### F1 Model

    perception/fusion/f1_hybrid.py

Recovered F1 hybrid architecture.

F1 combines:

- SPVCNN point/sparse features
- Circular / Range-View Neural Branch
- point-domain fusion
- 24-class classifier

### Canonical Preprocessing

    perception/sample_pipeline.py
    perception/geometry/
    perception/spvcnn/
    perception/circular/

These implement the preprocessing contract used by F1.

---

## CARLA LiDAR Contract

Standard CARLA LiDAR raw_data is expected as packed float32:

    x, y, z, intensity
    x, y, z, intensity
    ...

Default stride:

    4

The existing:

    mapping/carla_sensor_interface.py

continues to expose its original XYZ-only mapping contract.

The F1 integration separately preserves intensity in:

    perception/carla_pipeline.py

This avoids changing the established mapping adapter.

---

## F1 Input

Per-point feature vector:

    [x, y, z, intensity, range, azimuth, elevation]

Input channels:

    7

Voxel size:

    0.10 m

Unified semantic classes:

    24

Model prediction domain:

    0..23

Unified semantic domain:

    1..24

Unified label 0 is reserved for ignore.

---

## F1 Checkpoint

The selected checkpoint is intentionally NOT stored in Git.

Expected external checkpoint:

    models/f1/best_val_miou.pt

Selected training step:

    40000

Recorded validation mIoU:

    0.29713255014976925

Configure the actual checkpoint path on the machine running F1.

Do not commit:

    *.pt
    *.pth
    *.ckpt
    *.onnx

---

## F1 Runtime Requirement

The recovered canonical neural runtime was:

    OS: Linux x86-64
    Python: 3.10.12
    PyTorch: 2.7.1+cu128
    CUDA: 12.8
    TorchSparse: 2.1.0
    GPU used during recovered runtime: NVIDIA RTX 3050 6GB

The preserved TorchSparse binary is Linux/CPython-3.10 specific.

Therefore the current Windows development environment can test:

- preprocessing
- contracts
- CARLA parsing
- semantic adaptation
- SensorFrame integration
- MappingBackend integration
- WebSocket/backend regression

but it should NOT be treated as a validated environment for real
TorchSparse F1 neural inference.

Use a compatible Linux/NVIDIA environment for the actual F1 model.

---

## Semantic Mapping Warning

F1 outputs the project's unified 24-class semantic IDs.

FoveaMap currently consumes the following coarse semantic classes:

    0 UNKNOWN
    1 DRIVABLE
    2 NON_DRIVABLE
    3 STATIC_OBSTACLE
    4 VEHICLE
    5 VULNERABLE_USER

The authoritative historical mapping from unified classes 1..24 to
these six FoveaMap classes has NOT been recovered.

For safety:

    unmapped F1 class -> UNKNOWN (0)

The integration therefore does NOT invent a semantic taxonomy.

A verified mapping should be supplied through class_mapping when the
authoritative 24-class class definitions are available.

Do not interpret UNKNOWN fallback operation as final semantic
performance.

---

## CARLA Ground Truth

If CARLA semantic LiDAR is used, CARLA semantic tags should be retained
for evaluation/metrics.

They should NOT replace F1 predictions when demonstrating the trained
F1 perception pipeline.

The demonstration path should remain:

    raw CARLA LiDAR
        -> F1 prediction
        -> FoveaMap

not:

    CARLA ground-truth semantic labels
        -> FoveaMap

unless explicitly running a ground-truth/oracle evaluation.

---

## Basic Python Integration

Conceptually:

    from perception.inference import F1InferenceEngine
    from perception.carla_pipeline import CarlaF1MappingPipeline

    engine = F1InferenceEngine(
        checkpoint_path="PATH_TO_CHECKPOINT",
        device="cuda",
    )

    engine.load()

    pipeline = CarlaF1MappingPipeline(
        perception_engine=engine,
        class_mapping=VERIFIED_CLASS_MAPPING,
    )

For every CARLA LiDAR callback:

    output = pipeline.process(
        measurement=lidar_measurement,
        vehicle=ego_vehicle,
    )

`output` is the existing BackendOutput object.

The existing backend/WebSocket serialization should be used for the
frontend rather than creating a second frontend schema.

---

## Testing

Integration-only tests:

    python -m pytest tests/test_perception_inference.py tests/test_carla_f1_pipeline.py -q

Expected:

    13 passed

Full repository regression:

    python -m pytest -q

Current verified result:

    832 passed

---

## Tests Added

    tests/test_perception_inference.py
    tests/test_carla_f1_pipeline.py

They verify:

- F1 deployment preprocessing contract
- SparseBatch construction
- model/unified label domains
- CARLA intensity preservation
- 24 -> 6 semantic adapter mechanics
- semantic confidence propagation
- CARLA metadata propagation
- CARLA numeric frame ID normalization
- SensorFrame construction
- MappingBackend compatibility

No CARLA installation is required for these contract tests.

---

## What Aryan Needs To Connect

Aryan's CARLA/frontend side needs to provide:

1. CARLA LiDAR measurement object

       measurement.raw_data
       measurement.frame
       measurement.timestamp

2. Ego CARLA vehicle actor supporting:

       get_velocity()
       get_angular_velocity()
       get_transform()

3. Compatible F1 inference runtime

4. External F1 checkpoint

5. Verified 24-class -> 6-class semantic mapping when available

The repository already handles the remaining integration boundary.

---

## Do Not Commit

Keep these outside Git:

- model checkpoints
- datasets
- raw CARLA captures
- generated runtime data
- secrets
- machine-specific environment files

The repository .gitignore has been updated accordingly.

---

## Known Remaining Deployment Tasks

These are deployment/environment tasks, not missing backend architecture:

1. Run F1 under compatible Linux/NVIDIA/TorchSparse environment.
2. Supply the selected F1 checkpoint externally.
3. Supply the authoritative unified-24 -> FoveaMap-6 mapping.
4. Connect Aryan's live CARLA LiDAR callback to CarlaF1MappingPipeline.
5. Send the resulting BackendOutput through the existing WebSocket/frontend path.
6. Optionally compare F1 predictions against CARLA semantic ground truth.

The mapping/backend implementation should not need to be rewritten.
