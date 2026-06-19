#!/usr/bin/env python3
"""Launch the RTM real-time slice streaming API server."""

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seismic_tomography.api_server import SliceStreamAPI


def main():
    parser = argparse.ArgumentParser(description="RTM Snapshot Streaming API Server")
    parser.add_argument("--mmap", type=str, required=True,
                        help="Path to RTM snapshots mmap file")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind host")
    parser.add_argument("--port", type=int, default=8765,
                        help="Bind port")
    parser.add_argument("--poll-interval", type=float, default=0.05,
                        help="File refresh poll interval (seconds)")
    args = parser.parse_args()

    print(f"[RTM API] Starting slice streaming server")
    print(f"[RTM API]   mmap: {args.mmap}")
    print(f"[RTM API]   HTTP: http://{args.host}:{args.port}")
    print(f"[RTM API]   WS:   ws://{args.host}:{args.port}/ws/stream")
    print(f"[RTM API]   Health: http://{args.host}:{args.port}/health")

    server = SliceStreamAPI(
        mmap_path=args.mmap,
        host=args.host,
        port=args.port,
        poll_interval=args.poll_interval,
    )

    server.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[RTM API] Shutting down...")
        server.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
