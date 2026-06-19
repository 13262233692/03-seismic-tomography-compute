"""Visualization node client for RTM real-time slice streaming.

Two access modes:
 1. Direct mmap (zero-copy, recommended for same-node visualization)
 2. Remote API (HTTP/WebSocket, for cross-network visualization)
"""

import base64
import io
import time
import threading
import queue
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import requests
    import websockets
    import asyncio
    HAS_NETWORK_CLIENT = True
except ImportError:
    HAS_NETWORK_CLIENT = False

from .api_server import MmapSliceReader, MMAP_FILE_MAGIC


@dataclass
class SliceFrame:
    frame_index: int
    timestep: int
    axis: int
    slice_index: int
    dim0: int
    dim1: int
    forward: Optional[np.ndarray] = None
    adjoint: Optional[np.ndarray] = None
    image_condition: Optional[np.ndarray] = None
    received_at: float = 0.0


class DirectMmapClient:
    """Zero-copy direct mmap client for same-node visualization."""

    def __init__(self, mmap_path: str):
        self.mmap_path = Path(mmap_path)
        self.reader = MmapSliceReader(str(self.mmap_path))
        self._opened = False

    def open(self):
        if self._opened:
            return
        for _ in range(10):
            if self.reader.open():
                self._opened = True
                return
            time.sleep(0.1)
        raise FileNotFoundError(f"Could not open mmap: {self.mmap_path}")

    def close(self):
        if self._opened:
            self.reader.close()
            self._opened = False

    def frame_count(self) -> int:
        if not self._opened:
            self.open()
        return self.reader.frame_count()

    def latest_index(self) -> int:
        if not self._opened:
            self.open()
        return self.reader.latest_frame_index()

    def get_frame(self, idx: int) -> Optional[SliceFrame]:
        if not self._opened:
            self.open()
        data = self.reader.read_slice(idx)
        if not data:
            return None
        return self._to_frame(data)

    def get_latest(self) -> Optional[SliceFrame]:
        idx = self.latest_index()
        if idx < 0:
            return None
        return self.get_frame(idx)

    def poll_new_frames(self,
                         start_idx: int = 0,
                         timeout: float = 30.0) -> List[SliceFrame]:
        frames = []
        t0 = time.time()
        next_idx = start_idx
        while time.time() - t0 < timeout:
            latest = self.latest_index()
            while next_idx <= latest:
                f = self.get_frame(next_idx)
                if f:
                    frames.append(f)
                next_idx += 1
            if frames:
                return frames
            time.sleep(0.02)
        return frames

    def stream_forever(self,
                        callback: Callable[[SliceFrame], None],
                        axis: Optional[int] = None,
                        stop_event: Optional[threading.Event] = None,
                        max_fps: float = 30.0):
        if stop_event is None:
            stop_event = threading.Event()

        self.open()
        next_idx = self.latest_index() + 1
        min_interval = 1.0 / max(1.0, max_fps)

        while not stop_event.is_set():
            latest = self.latest_index()
            while next_idx <= latest and not stop_event.is_set():
                f = self.get_frame(next_idx)
                if f and (axis is None or f.axis == axis):
                    f.received_at = time.time()
                    callback(f)
                next_idx += 1
            time.sleep(min_interval)

    @staticmethod
    def _to_frame(data: Dict[str, Any]) -> SliceFrame:
        return SliceFrame(
            frame_index=data.get('frame_index', -1),
            timestep=data.get('timestep', 0),
            axis=data.get('axis', 0),
            slice_index=data.get('slice_index', 0),
            dim0=data.get('dim0', 0),
            dim1=data.get('dim1', 0),
            forward=data.get('forward'),
            adjoint=data.get('adjoint'),
            image_condition=data.get('image_condition'),
            received_at=time.time(),
        )


class RemoteAPIClient:
    """HTTP/WebSocket client for cross-network visualization nodes."""

    def __init__(self, base_url: str = "http://localhost:8765"):
        if not HAS_NETWORK_CLIENT:
            raise ImportError(
                "requests and websockets required: pip install requests websockets"
            )
        self.base_url = base_url.rstrip('/')
        self.ws_url = base_url.replace('http://', 'ws://') + "/ws/stream"
        self._session = requests.Session()

    def close(self):
        self._session.close()

    def health(self) -> Dict[str, Any]:
        r = self._session.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def frame_count(self) -> int:
        return int(self._session.get(f"{self.base_url}/frames/count").json()['count'])

    def get_frame_index(self) -> List[Dict[str, Any]]:
        return self._session.get(f"{self.base_url}/frames/index").json()

    def get_frame(self, frame_id: int) -> Optional[SliceFrame]:
        try:
            r = self._session.get(f"{self.base_url}/frames/{frame_id}")
            if r.status_code != 200:
                return None
            return self._parse_json_slice(r.json())
        except Exception:
            return None

    def get_latest(self, axis: Optional[int] = None) -> Optional[SliceFrame]:
        try:
            params = {}
            if axis is not None:
                params['axis'] = axis
            r = self._session.get(f"{self.base_url}/frames/latest", params=params)
            if r.status_code != 200:
                return None
            return self._parse_json_slice(r.json())
        except Exception:
            return None

    def stream_websocket(self,
                          callback: Callable[[SliceFrame], None],
                          axis: Optional[int] = None,
                          start_frame: Optional[int] = None,
                          max_fps: float = 30.0,
                          stop_event: Optional[threading.Event] = None):
        if stop_event is None:
            stop_event = threading.Event()

        async def _run():
            uri = self.ws_url
            params = []
            if axis is not None:
                params.append(f"axis={axis}")
            if start_frame is not None:
                params.append(f"start_frame={start_frame}")
            params.append(f"max_fps={max_fps}")
            if params:
                uri += "?" + "&".join(params)

            async with websockets.connect(uri) as ws:
                while not stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        import json
                        data = json.loads(msg)
                        if data.get('type') == 'slice':
                            pass
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break

        def _thread():
            asyncio.run(_run())

        t = threading.Thread(target=_thread, daemon=True)
        t.start()
        return t

    @staticmethod
    def _parse_json_slice(data: Dict[str, Any]) -> SliceFrame:
        f = SliceFrame(
            frame_index=data.get('frame_index', -1),
            timestep=data.get('timestep', 0),
            axis=data.get('axis', 0),
            slice_index=data.get('slice_index', 0),
            dim0=data.get('dim0', 0),
            dim1=data.get('dim1', 0),
            received_at=time.time(),
        )

        for key in ['forward', 'adjoint', 'image_condition']:
            b64_key = f"{key}_npy_b64"
            if b64_key in data:
                raw = base64.b64decode(data[b64_key])
                buf = io.BytesIO(raw)
                setattr(f, key if key != 'image_condition' else 'image_condition',
                        np.load(buf, allow_pickle=False))

        return f


class VisualizationClient:
    """Unified client that picks direct/remote mode automatically."""

    def __init__(self,
                 mmap_path: Optional[str] = None,
                 remote_url: Optional[str] = None):
        self.mmap_path = mmap_path
        self.remote_url = remote_url
        self._direct: Optional[DirectMmapClient] = None
        self._remote: Optional[RemoteAPIClient] = None

        if mmap_path and Path(mmap_path).exists():
            self._direct = DirectMmapClient(mmap_path)
            self._direct.open()
        elif remote_url:
            self._remote = RemoteAPIClient(remote_url)

    @property
    def mode(self) -> str:
        if self._direct:
            return "direct_mmap"
        if self._remote:
            return "remote_api"
        return "uninitialized"

    def close(self):
        if self._direct:
            self._direct.close()
        if self._remote:
            self._remote.close()

    def get_latest(self, axis: Optional[int] = None) -> Optional[SliceFrame]:
        if self._direct:
            return self._direct.get_latest()
        if self._remote:
            return self._remote.get_latest(axis=axis)
        return None

    def frame_count(self) -> int:
        if self._direct:
            return self._direct.frame_count()
        if self._remote:
            return self._remote.frame_count()
        return 0


def run_mmap_visualization_demo(mmap_path: str, axis: int = 2):
    """Simple terminal-based slice visualization demo (for sanity checking)."""
    client = DirectMmapClient(mmap_path)
    client.open()

    print("=== RTM Slice Stream Monitor (Ctrl+C to quit) ===")
    try:
        last_count = 0
        while True:
            count = client.frame_count()
            if count > last_count:
                latest = client.get_latest()
                if latest:
                    axis_name = {0: 'X', 1: 'Y', 2: 'Z'}.get(latest.axis, f"Axis{latest.axis}")
                    if latest.axis == axis or axis < 0:
                        fw_min = latest.forward.min() if latest.forward is not None else 0
                        fw_max = latest.forward.max() if latest.forward is not None else 0
                        ic_min = latest.image_condition.min() if latest.image_condition is not None else 0
                        ic_max = latest.image_condition.max() if latest.image_condition is not None else 0
                        print(
                            f"[t={latest.timestep:4d}] Frame#{latest.frame_index:4d} "
                            f"axis={axis_name}@{latest.slice_index} "
                            f"forward=[{fw_min:+.2e},{fw_max:+.2e}] "
                            f"IC=[{ic_min:+.2e},{ic_max:+.2e}] "
                            f"total={count}"
                        )
                last_count = count
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RTM visualization client demo")
    parser.add_argument("--mmap", type=str, required=True, help="RTM snapshots mmap file")
    parser.add_argument("--axis", type=int, default=-1, help="Show only this axis (0=X, 1=Y, 2=Z, -1=all)")
    args = parser.parse_args()
    run_mmap_visualization_demo(args.mmap, args.axis)
