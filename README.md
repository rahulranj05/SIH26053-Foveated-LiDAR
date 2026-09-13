# FoveaMap

## Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

**SIH Problem Statement:** SIH26053
**Project:** FoveaMap
**Branch:** `advit-mapping`

FoveaMap is a context-aware LiDAR mapping system that converts raw 3D LiDAR point clouds into a **hierarchical, variable-resolution 2.5D map**.

Instead of processing every region at the same resolution, FoveaMap allocates finer spatial resolution where perception matters most and coarser resolution where detail is less critical.

The system combines:

* Distance-based foveation
* Vehicle kinematics
* Predicted path relevance
* Semantic importance
* Dynamic-object information
* Hierarchical 2.5D mapping
* Temporal frame processing
* Ego-motion compensation
* Safety-aware degradation
* Live visualization and dashboarding

The final system has been validated through **784 automated tests**.

---

# 1. Problem

Traditional LiDAR perception faces a fundamental trade-off:

**High resolution → high computational and memory cost**

**Low resolution → loss of important environmental detail**

A vehicle does not need the same mapping precision everywhere.

For example:

* A pedestrian 5 m ahead deserves high-resolution representation.
* A distant road region can tolerate coarser representation.
* A moving vehicle deserves more attention than an irrelevant static region.
* A region outside the predicted vehicle path may require less detail.

FoveaMap addresses this by creating a **perception fovea** around important regions.

---

# 2. FoveaMap Concept

The system begins with a LiDAR point cloud:

```text
Raw LiDAR Point Cloud
        |
        v
SensorFrame
        |
        v
Semantic / Dynamic / Motion Signals
        |
        v
Unified Foveation Controller
        |
        v
Distance + Context-aware Resolution
        |
        v
Hierarchical Leaf Mapper
        |
        v
2.5D Foveated Map
        |
        v
Safety Controller
        |
        v
Dashboard / Visualization
```

The central principle is:

> **Resolution follows perception importance.**

Distance remains the fundamental spatial rule, while contextual signals refine where computational attention should be concentrated.

---

# 3. Adaptive Resolution

The base distance hierarchy is:

| Distance | Target Resolution |
| -------- | ----------------: |
| 0–10 m   |              5 cm |
| 10–25 m  |             10 cm |
| 25–50 m  |             20 cm |
| 50–100 m |             50 cm |

The implementation uses a strict hierarchical refinement structure:

```text
40 cm
  |
  +-- 20 cm
        |
        +-- 10 cm
              |
              +-- 5 cm
```

This guarantees integer refinement between levels and avoids incompatible 50/20 cm parent-child relationships.

Only final leaf cells are retained.

This prevents ancestor/descendant overlap and preserves point conservation.

---

# 4. Context-Aware Foveation

FoveaMap combines multiple perception signals.

### Distance

Nearby regions receive greater spatial importance.

### Kinematics

Vehicle speed and motion influence the useful forward perception range.

### Predicted Path

Regions near the predicted vehicle trajectory receive additional attention.

### Semantics

The canonical semantic ontology is:

```text
UNKNOWN          = 0
DRIVABLE         = 1
NON_DRIVABLE     = 2
STATIC_OBSTACLE  = 3
VEHICLE          = 4
VULNERABLE_USER  = 5
```

Default semantic importance:

```text
UNKNOWN          0.35
DRIVABLE         0.00
NON_DRIVABLE     0.40
STATIC_OBSTACLE  0.75
VEHICLE          0.85
VULNERABLE_USER  1.00
```

### Dynamic Information

Moving objects receive additional perception priority.

Temporal motion evidence can be generated from consecutive LiDAR frames, with ego-motion compensation available when vehicle motion is supplied.

---

# 5. Unified Foveation

The unified controller combines:

```text
DISTANCE
KINEMATIC
PREDICTED_PATH
SEMANTIC
DYNAMIC
```

The controller produces:

```text
importance
resolution
dominant_reason
```

The fusion strategy is deliberately conservative: the strongest applicable importance signal determines the unified importance.

This ensures that a safety-critical object cannot become low-resolution merely because another subsystem assigns it a lower score.

---

# 6. Hierarchical 2.5D Mapping

Each final map leaf stores terrain and perception information such as:

* Elevation statistics
* Minimum / maximum / mean / variance of Z
* Slope
* Roughness
* Discontinuity
* Semantic class
* Semantic confidence
* Traversability
* Dynamic probability
* Risk
* Timestamp
* Ground elevation
* Maximum obstacle height
* Vertical clearance

The result is a compact **2.5D semantic-terrain representation** rather than a uniformly dense 3D point cloud.

---

# 7. Temporal Processing

FoveaMap supports sequential LiDAR frames.

The temporal pipeline provides:

```text
Frame N-1
    |
    | ego-motion compensation
    v
Aligned previous cloud
    |
    +--------+
             |
Frame N ----> Motion evidence
             |
             v
       Dynamic signal
             |
             v
       FoveaMap pipeline
```

Supported temporal components include:

* Sequential dataset processing
* Frame chronology validation
* Timestamp handling
* Ego-motion transformation
* Previous/current frame compensation
* Temporal dynamic estimation
* Explicit dynamic-probability integration
* Semantic signal integration

---

# 8. Safety Controller

FoveaMap includes three operating modes:

```text
NORMAL
   |
   v
DEGRADED
   |
   v
SAFETY
```

The controller monitors conditions such as:

* Point count
* Invalid-point fraction
* Processing latency
* Frame staleness

When perception quality deteriorates, the system reduces the usable mapping range.

Default behavior:

| Mode     |         Maximum Range |
| -------- | --------------------: |
| NORMAL   | Full configured range |
| DEGRADED |                  50 m |
| SAFETY   |                  25 m |

This creates an explicit fallback mechanism rather than allowing degraded perception to silently produce an apparently normal map.

---

# 9. Dataset and Sensor Interfaces

The architecture supports a sensor-independent `SensorFrame` abstraction containing:

```text
XYZ point cloud
Vehicle state
Timestamp
Frame ID
Optional semantic labels
Optional dynamic probabilities
```

Implemented interfaces include:

* SemanticKITTI
* Sequential LiDAR datasets
* CARLA sensor interface
* CARLA dynamic-world scenarios

The system was designed so that the mapping pipeline is independent of a particular physical LiDAR manufacturer.

---

# 10. Evaluation

FoveaMap includes evaluation infrastructure for:

### Terrain Accuracy

Metrics include:

* Elevation RMSE
* Terrain-specific RMSE
* Coverage
* Valid-cell statistics

Current synthetic terrain benchmark:

```text
Flat terrain RMSE       0.015206 m
Linear slope RMSE       0.010976 m
Stepped terrain RMSE    0.010108 m
Uneven terrain RMSE     0.011210 m
Coverage                100%
```

### Dynamic Object Detection

Current synthetic benchmark:

```text
Precision    1.0000
Recall       1.0000
F1           1.0000
FPR          0.0000

TP = 11
FP = 0
TN = 11
FN = 0
```

These are controlled benchmark results and should not be interpreted as comprehensive real-world accuracy.

### Baselines

The project contains comparisons against:

* Uniform 2.5D
* Uniform 3D
* Distance-only adaptive mapping
* FoveaMap

The current benchmark demonstrates the adaptive mapping behavior, but FoveaMap is **not claimed to be faster than every baseline**.

### Ablation

Implemented ablations include:

```text
Distance only
Distance + Kinematic
Distance + Path
Distance + Semantic
Distance + Dynamic
Full FoveaMap
```

### Temporal Optimization

Temporal correspondence was optimized using a dependency-free spatial hashing approach.

At the current 124,668-point benchmark scale:

```text
Mean       ≈ 1107 ms
Median     ≈ 1102 ms
P95        ≈ 1131 ms
Update     ≈ 0.90 FPS
```

This is an important current limitation of the CPU-oriented prototype and is reported honestly rather than hidden.

---

# 11. Live Demonstration

The final demonstration produces a synthetic LiDAR stream and runs it through the complete architecture.

Run:

```bash
python3 scripts/run_final_demo.py
```

For a headless demonstration:

```bash
python3 scripts/run_final_demo.py --frames 60 --fps 10 --seed 39 --no-show
```

Short smoke test:

```bash
python3 scripts/run_final_demo.py --frames 6 --fps 10 --no-show
```

Available options:

```text
--frames
--fps
--seed
--no-show
--save-dir
```

The packaged 60-frame demonstration intentionally transitions through:

```text
Frames 0–19    NORMAL
Frames 20–39   DEGRADED
Frames 40–49   SAFETY
Frames 50–59   NORMAL recovery
```

This demonstrates that the safety controller responds dynamically to changing perception conditions.

---

# 12. Dashboard

The final dashboard exposes:

* Current frame
* Point count
* Valid / invalid points
* Safety mode
* Safety reasons
* Leaf count
* Resolution distribution
* Dominant foveation reason
* Processing latency
* FPS
* LiDAR preview

The visual dashboard is a presentation layer over the existing FoveaMap pipeline; it does not contain a separate mapping implementation.

---

# 13. Project Architecture

```text
mapping/
├── sensor_frame.py
├── semantic_kitti_adapter.py
├── sequential_dataset.py
├── sequential_processor.py
├── ego_motion.py
├── temporal_motion.py
├── signal_integration.py
├── temporal_foveamap_pipeline.py
├── foveamap_pipeline.py
├── foveamap_dashboard.py
├── foveamap_visual_dashboard.py
├── safety_controller.py
├── baselines.py
└── ...

evaluation/
├── terrain_accuracy.py
├── dynamic_object_metrics.py
├── baseline_metrics.py
└── ablation.py

scripts/
├── run_foveamap_dashboard.py
├── run_final_demo.py
├── a40_final_validation.py
├── a40_benchmark_consolidation.py
└── benchmark_*.py

tests/
└── comprehensive automated regression suite
```

---

# 14. Development Milestones

The project was developed incrementally:

```text
A1–A4    Basic 2D mapping
A5       Real SemanticKITTI frame
A6       Terrain features
A7       Traversability
A8       Distance-adaptive resolution
A9       Hierarchical grid
A10–A12 Benchmarking and vectorization
A13      Kinematic foveation
A14      Predicted-path foveation
A15      Semantic foveation
A16      Dynamic foveation
A17      Unified foveation controller
A18      Hierarchical foveated mapper
A19      Behaviour validation
A20      Complete FoveaMap pipeline
A21      SensorFrame interface
A22      SemanticKITTI adapter
A23      Sequential frame processor
A24      2.5D visualization
A25      Temporal motion integration
A26      Ego-motion compensation
A27      CARLA sensor interface
A28      Dynamic CARLA scenarios
A29      Sequential dataset integration
A30      Semantic + dynamic signal integration
A31      Temporal FoveaMap pipeline
A32      Dataset FoveaMap runner
A33      Terrain accuracy evaluation
A34      Dynamic-object evaluation
A35      Baseline comparison
A36      Ablation studies
A37      Performance optimization
A38      Robustness and safety modes
A39      Final dashboard and live demonstration
A40      Final validation, benchmarking and demo packaging
```

---

# 15. Validation

The final repository currently passes:

```text
784 passed
```

The final validation framework checks:

* Required project files
* Benchmark inventory
* Core module imports
* Live dashboard tests
* Full regression suite

The final demonstration has also been exercised through all four operating phases:

```text
NORMAL
DEGRADED
SAFETY
NORMAL recovery
```

---

# 16. Reproducibility

Run the complete test suite:

```bash
python3 -m pytest -q
```

Run final validation:

```bash
python3 scripts/a40_final_validation.py
```

Run benchmark consolidation:

```bash
python3 scripts/a40_benchmark_consolidation.py
```

Run the final demonstration:

```bash
python3 scripts/run_final_demo.py --frames 60 --fps 10 --seed 39 --no-show
```

---

# 17. Current Limitations

FoveaMap is currently a research/prototype implementation rather than a production autonomous-driving stack.

Important limitations include:

1. The final live demonstration currently uses synthetic LiDAR rather than a physical LiDAR sensor.
2. Some evaluation results are controlled synthetic benchmarks.
3. The current CPU-oriented temporal correspondence stage is not yet real-time at full SemanticKITTI frame size.
4. Hardware acceleration, deployment optimization and embedded automotive integration remain future work.
5. Baseline performance varies by workload; FoveaMap should not be presented as universally faster than uniform or distance-only approaches.
6. The current system demonstrates the complete perception architecture and safety behavior, but additional real-world testing is required before safety-critical deployment.

These limitations are deliberately documented to distinguish demonstrated engineering capability from claims requiring further validation.

---

# 18. Future Work

Potential next steps include:

* GPU / CUDA acceleration
* OpenVINO deployment
* Sparse convolution backends
* Real-time LiDAR sensor integration
* Larger real-world sequential datasets
* Indian road-scene datasets
* Improved dynamic-object tracking
* Learned foveation policies
* Hardware-in-the-loop testing
* Embedded automotive deployment
* Larger-scale real-world safety validation

---

# 19. Summary

FoveaMap addresses the computational challenge of dense LiDAR perception by replacing uniform spatial processing with **adaptive, hierarchical, context-aware resolution**.

Its key contribution is not simply reducing the number of cells.

It combines:

```text
DISTANCE
   +
KINEMATICS
   +
PREDICTED PATH
   +
SEMANTICS
   +
DYNAMIC INFORMATION
   +
TEMPORAL CONTEXT
   +
SAFETY STATE
```

to determine **where perception needs detail most**.

The resulting architecture provides:

* Adaptive 2.5D mapping
* Hierarchical spatial resolution
* Semantic awareness
* Dynamic-object awareness
* Temporal processing
* Ego-motion compensation
* Safety-aware degradation
* Reproducible evaluation
* Live visualization
* Automated validation

**FoveaMap turns LiDAR mapping from a uniform processing problem into a perception-prioritized mapping problem.**
