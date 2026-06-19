import os
import io
import time
import struct
import asyncio
import threading
import mmap
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
    from fastapi.responses import StreamingResponse, JSONResponse, Response
    from pydantic import BaseModel
    import numpy as np
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

MMAP_FILE_MAGIC = 0x52544D534E4150
SLICE_HEADER_SIZE = 128
FRAME_MAGIC = 0x534C494345303031


@dataclass
class FrameIndexEntry:
    frame_index: int
    timestep: int
    axis: int
    slice_index: int
    dim0: int
    dim1: int
    file_offset: int
    frame_size: int


class MmapSliceReader:
    def __init__(self, mmap_path: str):
        self.mmap_path = Path(mmap_path)
        self._mm = None
        self._fh = None
        self._last_mtime = 0
        self._frame_count_cache = 0
        self._index_cache: List[FrameIndexEntry] = []
        self._lock = threading.RLock()

    def open(self):
        if not self.mmap_path.exists():
            return False
        with self._lock:
            self._fh = open(str(self.mmap_path), 'rb')
            file_size = os.fstat(self._fh.fileno()).st_size
            self._mm = mmap.mmap(self._fh.fileno(), length=0, access=mmap.ACCESS_READ)
            self._rebuild_index()
            self._last_mtime = self.mmap_path.stat().st_mtime
        return True

    def close(self):
        with self._lock:
            if self._mm:
                self._mm.close()
                self._mm = None
            if self._fh:
                self._fh.close()
                self._fh = None
            self._index_cache.clear()

    def _check_refresh(self):
        if not self.mmap_path.exists():
            return
        try:
            mt = self.mmap_path.stat().st_mtime
            if mt > self._last_mtime:
                self._last_mtime = mt
                self._rebuild_index()
        except OSError:
            pass

    def _rebuild_index(self):
        if not self._mm:
            return
        self._index_cache.clear()
        try:
            if len(self._mm) < 128:
                return
            magic = struct.unpack_from('<Q', self._mm, 0)[0]
            if magic != MMAP_FILE_MAGIC:
                return
            write_offset = struct.unpack_from('<q', self._mm, 32)[0]
            frame_count = struct.unpack_from('<q', self._mm, 24)[0]
            self._frame_count_cache = int(frame_count)

            offset = 128
            frame_idx = 0
            while offset + 8 <= int(write_offset) and frame_idx < int(frame_count) + 100:
                if offset + 8 > len(self._mm):
                    break
                frame_size = struct.unpack_from('<q', self._mm, offset)[0]
                offset += 8

                if frame_size <= 0 or frame_size > 100000000:
                    break
                if offset + frame_size > len(self._mm):
                    break

                if frame_size >= SLICE_HEADER_SIZE:
                    slice_magic = struct.unpack_from('<Q', self._mm, offset)[0]
                    if slice_magic == FRAME_MAGIC:
                        timestep = struct.unpack_from('<q', self._mm, offset + 8)[0]
                        axis = struct.unpack_from('<i', self._mm, offset + 16)[0]
                        sl_idx = struct.unpack_from('<q', self._mm, offset + 20)[0]
                        d0 = struct.unpack_from('<q', self._mm, offset + 28)[0]
                        d1 = struct.unpack_from('<q', self._mm, offset + 36)[0]

                        self._index_cache.append(FrameIndexEntry(
                            frame_index=frame_idx,
                            timestep=int(timestep),
                            axis=int(axis),
                            slice_index=int(sl_idx),
                            dim0=int(d0),
                            dim1=int(d1),
                            file_offset=offset,
                            frame_size=int(frame_size),
                        ))

                offset += frame_size
                frame_idx += 1

        except Exception:
            pass

    def frame_count(self) -> int:
        self._check_refresh()
        with self._lock:
            return len(self._index_cache)

    def latest_frame_index(self) -> int:
        self._check_refresh()
        with self._lock:
            return len(self._index_cache) - 1 if self._index_cache else -1

    def read_slice_bytes(self, frame_idx: int) -> Optional[bytes]:
        self._check_refresh()
        with self._lock:
            if frame_idx < 0 or frame_idx >= len(self._index_cache):
                return None
            entry = self._index_cache[frame_idx]
            if entry.file_offset + entry.frame_size > len(self._mm):
                return None
            return bytes(self._mm[entry.file_offset:entry.file_offset + entry.frame_size])

    def read_slice(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        raw = self.read_slice_bytes(frame_idx)
        if raw is None or len(raw) < SLICE_HEADER_SIZE:
            return None
        try:
            magic = struct.unpack_from('<Q', raw, 0)[0]
            if magic != FRAME_MAGIC:
                return None
            timestep = struct.unpack_from('<q', raw, 8)[0]
            axis = struct.unpack_from('<i', raw, 16)[0]
            sl_idx = struct.unpack_from('<q', raw, 20)[0]
            d0 = struct.unpack_from('<q', raw, 28)[0]
            d1 = struct.unpack_from('<q', raw, 36)[0]
            fw_off = struct.unpack_from('<q', raw, 72)[0]
            adj_off = struct.unpack_from('<q', raw, 80)[0]
            ic_off = struct.unpack_from('<q', raw, 88)[0]
            fw_bytes = struct.unpack_from('<q', raw, 96)[0]
            adj_bytes = struct.unpack_from('<q', raw, 104)[0]
            ic_bytes = struct.unpack_from('<q', raw, 112)[0]

            n_fw = fw_bytes // 8
            n_adj = adj_bytes // 8
            n_ic = ic_bytes // 8

            result = {
                'frame_index': frame_idx,
                'timestep': int(timestep),
                'axis': int(axis),
                'slice_index': int(sl_idx),
                'dim0': int(d0),
                'dim1': int(d1),
            }

            if n_fw > 0 and fw_off + fw_bytes <= len(raw):
                fw = np.frombuffer(raw[fw_off:fw_off + fw_bytes], dtype=np.float64)
                result['forward'] = fw.reshape(d0, d1).copy()
            if n_adj > 0 and adj_off + adj_bytes <= len(raw):
                adj = np.frombuffer(raw[adj_off:adj_off + adj_bytes], dtype=np.float64)
                result['adjoint'] = adj.reshape(d0, d1).copy()
            if n_ic > 0 and ic_off + ic_bytes <= len(raw):
                ic = np.frombuffer(raw[ic_off:ic_off + ic_bytes], dtype=np.float64)
                result['image_condition'] = ic.reshape(d0, d1).copy()

            return result
        except Exception:
            return None

    def get_index(self) -> List[Dict[str, Any]]:
        self._check_refresh()
        with self._lock:
            return [
                {
                    'frame_index': e.frame_index,
                    'timestep': e.timestep,
                    'axis': e.axis,
                    'slice_index': e.slice_index,
                    'dim0': e.dim0,
                    'dim1': e.dim1,
                }
                for e in self._index_cache
            ]


if not HAS_FASTAPI:
    class SliceStreamAPI:
        def __init__(self, mmap_path: str, host: str = "0.0.0.0", port: int = 8765):
            self.mmap_path = mmap_path
            self.host = host
            self.port = port

        def start(self):
            raise ImportError(
                "FastAPI is not installed. "
                "Run: pip install fastapi uvicorn websockets numpy"
            )
else:
    class SliceStreamAPI:
        def __init__(self, mmap_path: str, host: str = "0.0.0.0", port: int = 8765,
                     poll_interval: float = 0.05):
            self.mmap_path = mmap_path
            self.host = host
            self.port = port
            self.poll_interval = poll_interval
            self.reader = MmapSliceReader(mmap_path)
            self._stop_event = threading.Event()
            self._server_thread: Optional[threading.Thread] = None

            self.app = FastAPI(
                title="RTM Snapshot Streaming API",
                description="Real-time reverse-time migration slice streaming service",
                version="1.1.0",
            )

            self._setup_routes()

        def _setup_routes(self):
            app = self.app

            @app.get("/health")
            async def health():
                return {
                    "status": "ok",
                    "mmap_file": self.mmap_path,
                    "file_exists": os.path.exists(self.mmap_path),
                    "total_frames": self.reader.frame_count(),
                    "latest_frame": self.reader.latest_frame_index(),
                }

            @app.get("/frames/count")
            async def frame_count():
                return {"count": self.reader.frame_count()}

            @app.get("/frames/index")
            async def frame_index():
                return self.reader.get_index()

            @app.get("/frames/latest")
            async def get_latest(axis: Optional[int] = Query(None, ge=0, le=3),
                                  index: Optional[int] = None):
                latest_idx = self.reader.latest_frame_index()
                if latest_idx < 0:
                    raise HTTPException(status_code=404, detail="No frames available")

                if axis is not None or index is not None:
                    idx = self.reader.get_index()
                    for entry in reversed(idx):
                        if (axis is None or entry['axis'] == axis) and \
                           (index is None or entry['slice_index'] == index):
                            sl = self.reader.read_slice(entry['frame_index'])
                            if sl:
                                return self._serialize_slice(sl)

                sl = self.reader.read_slice(latest_idx)
                if not sl:
                    raise HTTPException(status_code=404, detail="Frame not found")
                return self._serialize_slice(sl)

            @app.get("/frames/{frame_id}")
            async def get_frame(frame_id: int):
                sl = self.reader.read_slice(frame_id)
                if not sl:
                    raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found")
                return self._serialize_slice(sl)

            @app.get("/frames/{frame_id}/forward.png")
            async def get_forward_png(frame_id: int):
                return await self._slice_png(frame_id, 'forward')

            @app.get("/frames/{frame_id}/adjoint.png")
            async def get_adjoint_png(frame_id: int):
                return await self._slice_png(frame_id, 'adjoint')

            @app.get("/frames/{frame_id}/ic.png")
            async def get_ic_png(frame_id: int):
                return await self._slice_png(frame_id, 'image_condition')

            @app.websocket("/ws/stream")
            async def stream_slices(websocket: WebSocket,
                                     axis: Optional[int] = None,
                                     start_frame: Optional[int] = None,
                                     max_fps: float = 30.0):
                await websocket.accept()
                try:
                    if start_frame is None:
                        next_idx = self.reader.latest_frame_index() + 1
                    else:
                        next_idx = start_frame

                    min_interval = 1.0 / max(1.0, max_fps)

                    while True:
                        latest = self.reader.latest_frame_index()
                        while next_idx <= latest:
                            sl = self.reader.read_slice(next_idx)
                            if sl and (axis is None or sl.get('axis') == axis):
                                await websocket.send_json({
                                    "type": "slice",
                                    "frame_index": sl['frame_index'],
                                    "timestep": sl['timestep'],
                                    "axis": sl['axis'],
                                    "slice_index": sl['slice_index'],
                                    "dim0": sl['dim0'],
                                    "dim1": sl['dim1'],
                                    "forward_min": float(sl.get('forward', np.zeros(1)).min()) if 'forward' in sl else 0,
                                    "forward_max": float(sl.get('forward', np.zeros(1)).max()) if 'forward' in sl else 0,
                                    "adjoint_min": float(sl.get('adjoint', np.zeros(1)).min()) if 'adjoint' in sl else 0,
                                    "adjoint_max": float(sl.get('adjoint', np.zeros(1)).max()) if 'adjoint' in sl else 0,
                                })
                            next_idx += 1
                        await asyncio.sleep(min_interval)
                except WebSocketDisconnect:
                    pass

        async def _slice_png(self, frame_id: int, key: str):
            sl = self.reader.read_slice(frame_id)
            if not sl or key not in sl:
                raise HTTPException(status_code=404, detail="Not found")
            arr = sl[key]

            arr_norm = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-12)
            arr_norm = (arr_norm * 255).astype(np.uint8)

            try:
                from PIL import Image
                img = Image.fromarray(arr_norm, mode='L')
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                return Response(content=buf.getvalue(), media_type="image/png")
            except ImportError:
                buf = io.BytesIO()
                buf.write(b"\x89PNG\r\n\x1a\n")
                return Response(content=buf.getvalue(), media_type="image/png")

        def _serialize_slice(self, sl: Dict[str, Any]) -> Dict[str, Any]:
            result = {
                "frame_index": sl['frame_index'],
                "timestep": sl['timestep'],
                "axis": sl['axis'],
                "slice_index": sl['slice_index'],
                "dim0": sl['dim0'],
                "dim1": sl['dim1'],
            }
            for key in ['forward', 'adjoint', 'image_condition']:
                if key in sl:
                    arr = sl[key]
                    buf = io.BytesIO()
                    np.save(buf, arr, allow_pickle=False)
                    result[f"{key}_npy_b64"] = __import__('base64').b64encode(buf.getvalue()).decode('ascii')
                    result[f"{key}_min"] = float(arr.min())
                    result[f"{key}_max"] = float(arr.max())
                    result[f"{key}_mean"] = float(arr.mean())
            return result

        def start(self):
            try:
                import uvicorn
            except ImportError:
                raise ImportError("uvicorn not installed: pip install uvicorn")

            if not os.path.exists(self.mmap_path):
                Path(self.mmap_path).parent.mkdir(parents=True, exist_ok=True)
                self._create_empty_mmap()

            self.reader.open()

            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
            server = uvicorn.Server(config)

            def run():
                asyncio.run(server.serve())

            self._server_thread = threading.Thread(target=run, daemon=True)
            self._server_thread.start()
            print(f"[RTM API] Slice stream API started on http://{self.host}:{self.port}")

        def stop(self):
            self._stop_event.set()
            self.reader.close()

        def _create_empty_mmap(self):
            fh = open(self.mmap_path, 'wb')
            fh.write(b'\x00' * 128)
            fh.close()
            fh = open(self.mmap_path, 'r+b')
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_WRITE)
            struct.pack_into('<Q', mm, 0, MMAP_FILE_MAGIC)
            struct.pack_into('<q', mm, 8, 128)
            file_size = os.fstat(fh.fileno()).st_size
            struct.pack_into('<q', mm, 16, file_size)
            struct.pack_into('<q', mm, 24, 0)
            struct.pack_into('<q', mm, 32, 128)
            mm.flush()
            mm.close()
            fh.close()

        @property
        def url(self) -> str:
            return f"http://{self.host}:{self.port}"

        @property
        def ws_url(self) -> str:
            return f"ws://{self.host}:{self.port}/ws/stream"
