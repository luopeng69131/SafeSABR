#!/usr/bin/env python3
"""Generate synthetic high-bitrate SABR video_size files.

This is a development scaffold for Starlink 4K/8K experiments. It avoids
blocking algorithm work on real DASH packaging while preserving realistic
chunk-size scale and mild VBR variation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_BITRATES_KBPS = [3000, 8000, 15000, 30000, 60000, 120000]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "safesabr" / "video_trace" / "video" / "starlink_4k8k_synth",
    )
    parser.add_argument("--chunk-duration-s", type=float, default=4.0)
    parser.add_argument("--chunks", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bitrates-kbps", type=int, nargs="+", default=DEFAULT_BITRATES_KBPS)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Shared temporal variation keeps all representations correlated while
    # still giving the simulator non-constant chunk sizes.
    phase = np.linspace(0, 2 * np.pi, args.chunks, endpoint=False)
    temporal = 1.0 + 0.10 * np.sin(phase) + 0.05 * rng.standard_normal(args.chunks)
    temporal = np.clip(temporal, 0.75, 1.25)

    for idx, bitrate_kbps in enumerate(args.bitrates_kbps):
        nominal_bytes = bitrate_kbps * 1000.0 * args.chunk_duration_s / 8.0
        sizes = np.maximum(1, np.rint(nominal_bytes * temporal)).astype(int)
        with (args.output_dir / f"video_size_{idx}").open("w") as f:
            for size in sizes:
                f.write(f"{size}\n")

    print(f"Wrote {len(args.bitrates_kbps)} video_size files to {args.output_dir}")
    print(f"Bitrate ladder kbps: {args.bitrates_kbps}")


if __name__ == "__main__":
    main()
