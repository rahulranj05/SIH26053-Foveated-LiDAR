import asyncio
import json

import websockets


async def main():
    uri = "ws://127.0.0.1:8000/ws"

    async with websockets.connect(uri) as websocket:
        # 1. Connection message
        message = await websocket.recv()
        connection = json.loads(message)

        print("Connection:", connection)

        assert connection["type"] == "connection"
        assert connection["status"] == "connected"
        assert connection["schema_version"] == "1.0"

        # 2. Send synthetic LiDAR frame
        payload = {
            "xyz": [
                [1.0, 0.0, 0.5],
                [1.2, 0.1, 0.6],
                [2.0, 0.2, 0.7],
                [5.0, 1.0, 1.0],
                [10.0, 2.0, 1.2],
                [20.0, 3.0, 1.5],
            ],
            "vehicle": {
                "speed": 5.0,
                "yaw_rate": 0.1,
                "heading": 0.0,
            },
            "timestamp": 2.0,
            "frame_id": "websocket-test-001",
        }

        await websocket.send(json.dumps(payload))

        # 3. Receive BackendOutput
        message = await websocket.recv()
        output = json.loads(message)

        print("Backend output:")
        print(json.dumps(output, indent=2))

        assert output["schema_version"] == "1.0"
        assert output["frame_id"] == "websocket-test-001"
        assert output["timestamp"] == 2.0
        assert output["map"]["input_points"] == 6
        assert output["map"]["active_cells"] == 6
        assert output["objects"] == []

        print("\nWebSocket test passed.")


if __name__ == "__main__":
    asyncio.run(main())