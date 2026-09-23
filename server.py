from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from backend.pipeline import MappingBackend
from backend.adapters.frontend_output import FrontendOutputAdapter


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="FoveaMap Backend Runtime",
    version="1.0",
)


# ============================================================
# RUNTIME STATE
# ============================================================

backend = MappingBackend()
serializer = FrontendOutputAdapter()

_start_time = time.time()
_last_frame_id: str | None = None
_last_timestamp: float | None = None


# ============================================================
# WEBSOCKET CLIENTS
# ============================================================

connected_clients: set[WebSocket] = set()


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    """Basic runtime health check."""

    return {
        "status": "ok",
        "service": "FoveaMap Backend Runtime",
        "schema_version": "1.0",
        "uptime_seconds": round(time.time() - _start_time, 3),
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/status")
def status() -> dict[str, Any]:
    """Return current backend runtime state."""

    return {
        "status": "running",
        "service": "FoveaMap Backend Runtime",
        "schema_version": "1.0",
        "last_frame_id": _last_frame_id,
        "last_timestamp": _last_timestamp,
        "has_previous_frame": backend.previous_frame is not None,
        "connected_clients": len(connected_clients),
    }


# ============================================================
# RESET
# ============================================================

@app.post("/reset")
def reset() -> dict[str, Any]:
    """Reset temporal backend state."""

    global _last_frame_id, _last_timestamp

    backend.reset()

    _last_frame_id = None
    _last_timestamp = None

    return {
        "status": "reset",
        "schema_version": "1.0",
    }


# ============================================================
# FRAME PROCESSING
# ============================================================

@app.post("/api/v1/frame")
def process_frame(payload: dict[str, Any]) -> JSONResponse:
    """
    Process one canonical SensorFrame.

    The server expects the transport layer to provide:
        xyz
        vehicle
        timestamp
        frame_id

    Semantic perception/model inference will be connected
    through the model-neutral inference adapter later.
    """

    global _last_frame_id, _last_timestamp

    try:
        frame = _build_sensor_frame(payload)

        output = backend.process(frame)

        _last_frame_id = str(output.frame_id)
        _last_timestamp = output.timestamp

        serialized = serializer.serialize(output)

        return JSONResponse(content=json.loads(serialized))

    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket transport for frontend/live runtime integration.
    """

    await websocket.accept()
    connected_clients.add(websocket)

    try:
        await websocket.send_json(
            {
                "type": "connection",
                "status": "connected",
                "schema_version": "1.0",
            }
        )

        while True:
            payload = await websocket.receive_json()

            try:
                frame = _build_sensor_frame(payload)

                output = backend.process(frame)
                serialized = serializer.serialize(output)

                await websocket.send_text(serialized)

            except Exception as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    except WebSocketDisconnect:
        pass

    finally:
        connected_clients.discard(websocket)


# ============================================================
# SENSOR FRAME CONSTRUCTION
# ============================================================

def _build_sensor_frame(payload: dict[str, Any]):
    """
    Convert transport JSON into the existing canonical SensorFrame.

    This function deliberately uses the existing mapping contract
    instead of creating a second SensorFrame implementation.
    """

    import numpy as np

    from mapping.kinematic_foveation import VehicleState
    from mapping.sensor_frame import SensorFrame

    if not isinstance(payload, dict):
        raise TypeError("Frame payload must be a JSON object")

    required = {"xyz", "vehicle", "timestamp", "frame_id"}
    missing = required - payload.keys()

    if missing:
        raise ValueError(
            f"Missing required frame fields: {sorted(missing)}"
        )

    vehicle_data = payload["vehicle"]

    if not isinstance(vehicle_data, dict):
        raise TypeError("vehicle must be an object")

    vehicle_required = {"speed", "yaw_rate", "heading"}
    missing_vehicle = vehicle_required - vehicle_data.keys()

    if missing_vehicle:
        raise ValueError(
            f"Missing vehicle fields: {sorted(missing_vehicle)}"
        )

    xyz = np.asarray(payload["xyz"], dtype=np.float64)

    vehicle = VehicleState(
        speed=float(vehicle_data["speed"]),
        yaw_rate=float(vehicle_data["yaw_rate"]),
        heading=float(vehicle_data["heading"]),
    )

    return SensorFrame(
        xyz=xyz,
        vehicle_state=vehicle,
        timestamp=float(payload["timestamp"]),
        frame_id=str(payload["frame_id"]),
    )


# ============================================================
# LOCAL STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )