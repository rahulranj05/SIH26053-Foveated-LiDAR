from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from backend.adapters.frontend_output import FrontendOutputAdapter
from backend.contracts.backend_output import BackendOutput
from perception.carla_pipeline import CarlaF1MappingPipeline
from perception.inference import F1InferenceEngine

app = FastAPI(title="FoveaMap Backend Runtime", version="1.0")
serializer = FrontendOutputAdapter()
connected_frontends: set[WebSocket] = set()
runtime_lock = asyncio.Lock()

_start_time = time.monotonic()
_last_frame_id: str | None = None
_last_timestamp: float | None = None
_last_error: str | None = None
_perception_engine: F1InferenceEngine | None = None
_pipeline: CarlaF1MappingPipeline | None = None
_model_loaded = False
_frames_processed = 0
_last_frame_wall_time: float | None = None
_frame_rate_ema = 0.0
_last_runtime: dict[str, Any] = {}
_last_source: dict[str, Any] = {}
_last_hardware_sample: dict[str, Any] = {}
_last_hardware_sample_wall: float = 0.0


def _checkpoint_path() -> Path:
    configured = os.environ.get("F1_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "models" / "f1" / "best_val_miou.pt").resolve()


def _dataset_id() -> str:
    return os.environ.get("F1_DATASET_ID", "SemanticKITTI")


def _device() -> str:
    return os.environ.get("F1_DEVICE", "cuda")


def _class_mapping() -> dict[int, int]:
    raw = os.environ.get("F1_BACKEND_CLASS_MAPPING_JSON", "{}").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("F1_BACKEND_CLASS_MAPPING_JSON must be a JSON object")
    return {int(key): int(value) for key, value in parsed.items()}


class _Vector:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _Rotation:
    def __init__(self, yaw: float):
        self.yaw = float(yaw)


class _Transform:
    def __init__(self, yaw: float):
        self.rotation = _Rotation(yaw)


class _Measurement:
    def __init__(self, raw_data: bytes, frame: str, timestamp: float):
        self.raw_data = raw_data
        self.frame = frame
        self.timestamp = float(timestamp)


class _Vehicle:
    def __init__(self, speed: float, yaw_rate_rad_s: float, heading_rad: float):
        self._velocity = _Vector(speed, 0.0, 0.0)
        self._angular_velocity = _Vector(0.0, 0.0, math.degrees(yaw_rate_rad_s))
        self._transform = _Transform(math.degrees(heading_rad))

    def get_velocity(self) -> _Vector:
        return self._velocity

    def get_angular_velocity(self) -> _Vector:
        return self._angular_velocity

    def get_transform(self) -> _Transform:
        return self._transform


def _ensure_pipeline() -> CarlaF1MappingPipeline:
    global _perception_engine, _pipeline, _model_loaded
    if _pipeline is not None:
        return _pipeline

    engine = F1InferenceEngine(_checkpoint_path(), device=_device())
    engine.load()
    _perception_engine = engine
    _pipeline = CarlaF1MappingPipeline(
        engine,
        class_mapping=_class_mapping(),
        dataset_id=_dataset_id(),
    )
    _model_loaded = True
    return _pipeline


def _build_raw_data(xyz: Any, intensity: Any) -> bytes:
    xyz_array = np.asarray(xyz, dtype=np.float32)
    intensity_array = np.asarray(intensity, dtype=np.float32)
    if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if intensity_array.shape != (xyz_array.shape[0],):
        raise ValueError("intensity must have shape (N,)")
    if not np.isfinite(xyz_array).all() or not np.isfinite(intensity_array).all():
        raise ValueError("LiDAR payload contains NaN or Infinity")
    raw = np.empty((xyz_array.shape[0], 4), dtype=np.float32)
    raw[:, :3] = xyz_array
    raw[:, 3] = intensity_array
    return raw.tobytes(order="C")


def _build_binary_frame(header: dict[str, Any], raw_data: bytes) -> BackendOutput:
    required = {"frame_id", "timestamp", "vehicle"}
    missing = required - header.keys()
    if missing:
        raise ValueError(f"Missing CARLA frame fields: {sorted(missing)}")
    vehicle = header["vehicle"]
    if not isinstance(vehicle, dict):
        raise TypeError("vehicle must be an object")
    for key in ("speed", "yaw_rate", "heading"):
        if key not in vehicle:
            raise ValueError(f"Missing vehicle field: {key}")

    measurement = _Measurement(raw_data, str(header["frame_id"]), float(header["timestamp"]))
    ego = _Vehicle(float(vehicle["speed"]), float(vehicle["yaw_rate"]), float(vehicle["heading"]))
    return _ensure_pipeline().process(measurement=measurement, vehicle=ego)


def _sample_hardware() -> dict[str, Any]:
    global _last_hardware_sample, _last_hardware_sample_wall
    now = time.monotonic()
    if now - _last_hardware_sample_wall < 1.0 and _last_hardware_sample:
        return _last_hardware_sample

    cpu_percent: float | None = None
    ram_percent: float | None = None
    try:
        import psutil
        cpu_percent = float(psutil.cpu_percent(interval=None))
        ram_percent = float(psutil.virtual_memory().percent)
    except Exception:
        pass

    gpu_util = None
    vram_used = None
    vram_total = None
    gpu_temp = None
    gpu_name = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=True,
        )
        row = completed.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 5:
            gpu_name = parts[0]
            gpu_util = float(parts[1])
            vram_used = float(parts[2])
            vram_total = float(parts[3])
            gpu_temp = float(parts[4])
    except Exception:
        pass

    _last_hardware_sample = {
        "cpu_percent": cpu_percent,
        "ram_percent": ram_percent,
        "gpu_util_percent": gpu_util,
        "vram_used_mb": vram_used,
        "vram_total_mb": vram_total,
        "gpu_temp_c": gpu_temp,
        "gpu_name": gpu_name,
    }
    _last_hardware_sample_wall = now
    return _last_hardware_sample


def _runtime_telemetry() -> dict[str, Any]:
    return {
        "type": "telemetry",
        "timestamp": time.time(),
        "frame_rate_hz": float(_frame_rate_ema),
        "frames_processed": int(_frames_processed),
        "points_processed": int(_last_runtime.get("points_processed", 0)),
        "active_cells": int(_last_runtime.get("active_cells", 0)),
        "backend_total_ms": float(_last_runtime.get("backend_total_ms", 0.0)),
        "controller_ms": float(_last_runtime.get("controller_ms", 0.0)),
        "mapper_ms": float(_last_runtime.get("mapper_ms", 0.0)),
        "foveation_points": int(_last_runtime.get("foveation_points", 0)),
        "delta_time": _last_runtime.get("delta_time"),
        "vehicle": _last_runtime.get("vehicle", {}),
        "source": _last_source,
        "hardware": _sample_hardware(),
    }


def _record_output(output: BackendOutput, source: dict[str, Any] | None = None) -> None:
    global _last_frame_id, _last_timestamp, _frames_processed, _last_frame_wall_time, _frame_rate_ema
    global _last_runtime, _last_source

    now = time.monotonic()
    if _last_frame_wall_time is not None:
        delta = now - _last_frame_wall_time
        if delta > 0:
            instant = 1.0 / delta
            _frame_rate_ema = instant if _frame_rate_ema <= 0 else (_frame_rate_ema * 0.85 + instant * 0.15)
    _last_frame_wall_time = now
    _frames_processed += 1
    _last_frame_id = str(output.frame_id)
    _last_timestamp = float(output.timestamp)
    _last_runtime = {
        "points_processed": int(output.map.get("input_points", 0)),
        "active_cells": int(output.map.get("active_cells", 0)),
        "backend_total_ms": float(output.metrics.get("total_ms", 0.0)),
        "controller_ms": float(output.metrics.get("controller_ms", 0.0)),
        "mapper_ms": float(output.metrics.get("mapper_ms", 0.0)),
        "foveation_points": int(output.metrics.get("foveation_points", 0)),
        "delta_time": output.metrics.get("delta_time"),
        "vehicle": output.vehicle,
    }
    if source is not None:
        _last_source = source


async def _broadcast(message: dict[str, Any] | str) -> None:
    stale: list[WebSocket] = []
    for websocket in list(connected_frontends):
        try:
            if isinstance(message, str):
                await websocket.send_text(message)
            else:
                await websocket.send_json(message)
        except Exception:
            stale.append(websocket)
    for websocket in stale:
        connected_frontends.discard(websocket)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "FoveaMap Backend Runtime",
        "schema_version": "1.0",
        "uptime_seconds": round(time.monotonic() - _start_time, 3),
        "model_loaded": _model_loaded,
        "model": "F1HybridModel",
        "checkpoint": str(_checkpoint_path()),
        "dataset_id": _dataset_id(),
        "semantic_mapping_entries": len(_class_mapping()),
        "last_error": _last_error,
        "connected_frontends": len(connected_frontends),
        "frames_processed": _frames_processed,
        "frame_rate_hz": round(_frame_rate_ema, 3),
        "hardware": _sample_hardware(),
        "source": _last_source,
    }


@app.get("/status")
def status() -> dict[str, Any]:
    return _runtime_telemetry() | {
        "status": "running",
        "service": "FoveaMap Backend Runtime",
        "schema_version": "1.0",
        "model_loaded": _model_loaded,
        "last_frame_id": _last_frame_id,
        "last_timestamp": _last_timestamp,
        "connected_frontends": len(connected_frontends),
        "last_error": _last_error,
    }


@app.post("/reset")
def reset() -> dict[str, Any]:
    global _last_frame_id, _last_timestamp, _frames_processed, _frame_rate_ema, _last_runtime, _last_source, _last_error
    if _pipeline is not None:
        _pipeline.reset()
    _last_frame_id = None
    _last_timestamp = None
    _frames_processed = 0
    _frame_rate_ema = 0.0
    _last_runtime = {}
    _last_source = {}
    _last_error = None
    return {"status": "reset", "schema_version": "1.0"}


@app.websocket("/ws")
async def frontend_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_frontends.add(websocket)
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "schema_version": "1.0",
            "model_loaded": _model_loaded,
            "model": "F1HybridModel",
            "checkpoint": str(_checkpoint_path()),
            "dataset_id": _dataset_id(),
            "semantic_mapping_entries": len(_class_mapping()),
            "source": _last_source,
        })
        while True:
            await websocket.send_json(_runtime_telemetry())
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        pass
    finally:
        connected_frontends.discard(websocket)


@app.websocket("/ws/ingest")
async def carla_ingest_websocket(websocket: WebSocket) -> None:
    global _last_error
    await websocket.accept()
    try:
        while True:
            header = json.loads(await websocket.receive_text())
            if not isinstance(header, dict) or header.get("type") != "carla_frame":
                raise ValueError("Expected a carla_frame header message")
            raw_data = await websocket.receive_bytes()

            async with runtime_lock:
                was_model_loaded = _model_loaded
                source = header.get("source") if isinstance(header.get("source"), dict) else None
                output = _build_binary_frame(header, raw_data)
                serialized = serializer.serialize(output)
                _record_output(output, source)
                _last_error = None

            if not was_model_loaded and _model_loaded:
                await _broadcast({
                    "type": "connection",
                    "status": "connected",
                    "schema_version": "1.0",
                    "model_loaded": True,
                    "model": "F1HybridModel",
                    "checkpoint": str(_checkpoint_path()),
                    "dataset_id": _dataset_id(),
                    "semantic_mapping_entries": len(_class_mapping()),
                    "source": _last_source,
                })

            await _broadcast(serialized)
            await _broadcast(_runtime_telemetry())
            await websocket.send_json({"type": "ack", "frame_id": output.frame_id, "timestamp": output.timestamp})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        try:
            await websocket.send_json({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        except Exception:
            pass


@app.post("/api/v1/frame")
def process_frame(payload: dict[str, Any]) -> JSONResponse:
    global _last_error
    try:
        raw_data = _build_raw_data(payload["xyz"], payload["intensity"])
        header = {
            "type": "carla_frame",
            "frame_id": str(payload["frame_id"]),
            "timestamp": float(payload["timestamp"]),
            "vehicle": payload["vehicle"],
            "source": payload.get("source", {}),
        }
        output = _build_binary_frame(header, raw_data)
        serialized = serializer.serialize(output)
        _record_output(output, header["source"])
        _last_error = None
        return JSONResponse(content=json.loads(serialized))
    except Exception as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        return JSONResponse(status_code=400, content={"status": "error", "error": type(exc).__name__, "message": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
