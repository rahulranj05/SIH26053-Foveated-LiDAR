from __future__ import annotations

from typing import Any

import numpy as np

from mapping.sensor_frame import SensorFrame
from mapping.temporal_foveamap_pipeline import (
    TemporalFoveaMapFrameResult,
    TemporalFoveaMapProcessor,
)

from backend.contracts.backend_output import BackendOutput


class MappingBackend:
    """
    Thin backend runtime adapter around the existing A31/A20/A18.3
    FoveaMap pipeline.

    This class does not modify the mapping implementation. It converts
    the existing TemporalFoveaMapFrameResult into the public BackendOutput
    contract.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        processor: TemporalFoveaMapProcessor | None = None,
    ) -> None:
        self.processor = (
            processor
            if processor is not None
            else TemporalFoveaMapProcessor()
        )

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def process(
        self,
        frame: SensorFrame,
    ) -> BackendOutput:
        """
        Process one canonical SensorFrame.

        Public backend flow:

            SensorFrame
                -> A31
                -> A20
                -> A18.3
                -> BackendOutput
        """

        if not isinstance(frame, SensorFrame):
            raise TypeError(
                "frame must be a SensorFrame"
            )

        result = self.processor.process_frame(frame)

        if not isinstance(
            result,
            TemporalFoveaMapFrameResult,
        ):
            raise TypeError(
                "TemporalFoveaMapProcessor.process_frame() "
                "must return a TemporalFoveaMapFrameResult"
            )

        return self._build_output(
            frame,
            result,
        )

    # ------------------------------------------------------------------
    # Backend output construction
    # ------------------------------------------------------------------

    def _build_output(
        self,
        frame: SensorFrame,
        result: TemporalFoveaMapFrameResult,
    ) -> BackendOutput:

        pipeline_result = result.pipeline_result
        leaf_map = pipeline_result.leaf_map
        foveation = pipeline_result.foveation
        timings = pipeline_result.timings_ms

        return BackendOutput(
            schema_version=self.SCHEMA_VERSION,
            frame_id=result.frame_id,
            timestamp=float(result.timestamp),

            vehicle=self._serialize_vehicle(
                frame
            ),

            map=self._serialize_map(
                leaf_map
            ),

            objects=[],

            metrics=self._serialize_metrics(
                result,
                timings,
                foveation,
            ),

            diagnostics=self._serialize_diagnostics(
                frame,
                leaf_map,
                result,
            ),
        )

    # ------------------------------------------------------------------
    # Vehicle
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_vehicle(
        frame: SensorFrame,
    ) -> dict[str, Any]:

        vehicle_state = frame.vehicle_state

        return {
            "speed": float(
                vehicle_state.speed
            ),
            "yaw_rate": float(
                vehicle_state.yaw_rate
            ),
            "heading": float(
                vehicle_state.heading
            ),
        }

    # ------------------------------------------------------------------
    # Hierarchical leaf map
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_map(
        leaf_map: Any,
    ) -> dict[str, Any]:

        return {
            "active_cells": int(
                leaf_map.active_cells
            ),

            "input_points": int(
                leaf_map.input_points
            ),

            "levels": (
                leaf_map.levels
                .astype(np.int8, copy=False)
                .tolist()
            ),

            "resolutions": (
                leaf_map.resolutions
                .astype(np.float64, copy=False)
                .tolist()
            ),

            "ix": (
                leaf_map.ix
                .astype(np.int64, copy=False)
                .tolist()
            ),

            "iy": (
                leaf_map.iy
                .astype(np.int64, copy=False)
                .tolist()
            ),

            "z_min": (
                leaf_map.z_min
                .astype(np.float64, copy=False)
                .tolist()
            ),

            "z_max": (
                leaf_map.z_max
                .astype(np.float64, copy=False)
                .tolist()
            ),

            "z_mean": (
                leaf_map.z_mean
                .astype(np.float64, copy=False)
                .tolist()
            ),

            "z_variance": (
                leaf_map.z_variance
                .astype(np.float64, copy=False)
                .tolist()
            ),

            "point_count": (
                leaf_map.point_count
                .astype(np.int64, copy=False)
                .tolist()
            ),

            "dominant_reason": [
                str(reason)
                for reason in leaf_map.dominant_reason
            ],
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_metrics(
        result: TemporalFoveaMapFrameResult,
        timings: Any,
        foveation: Any,
    ) -> dict[str, Any]:

        return {
            "controller_ms": float(
                timings["controller_ms"]
            ),

            "mapper_ms": float(
                timings["mapper_ms"]
            ),

            "total_ms": float(
                timings["total_ms"]
            ),

            "delta_time": (
                None
                if result.delta_time is None
                else float(result.delta_time)
            ),

            "foveation_points": int(
                foveation["importance"].size
            ),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_diagnostics(
        frame: SensorFrame,
        leaf_map: Any,
        result: TemporalFoveaMapFrameResult,
    ) -> dict[str, Any]:

        return {
            "backend": "MappingBackend",
            "pipeline": "A31_A20_A18.3",
            "input_points": int(
                frame.num_points
            ),
            "active_cells": int(
                leaf_map.active_cells
            ),
            "has_previous_frame": (
                result.delta_time is not None
            ),
            "signal_names": sorted(
                str(name)
                for name in result.signals.keys()
            ),
        }

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset temporal processing state."""

        self.processor.reset()

    @property
    def previous_frame(self) -> SensorFrame | None:
        """Return the processor's previous frame."""

        return self.processor.previous_frame
