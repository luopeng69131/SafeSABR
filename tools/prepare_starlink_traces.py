#!/usr/bin/env python3
"""Convert processed StarNet pickle files into SABR trace files.

SABR expects each trace file to contain two whitespace-separated columns:
relative_time_seconds and throughput_mbps.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


LOCATIONS = {
    "us": "US",
    "osn": "OSN",
    "vic": "VIC",
}


def install_pickle_compat() -> None:
    """Support older pandas pickle files that reference removed index modules."""
    module = types.ModuleType("pandas.core.indexes.numeric")
    module.Int64Index = pd.Index
    module.UInt64Index = pd.Index
    module.Float64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = module


def load_dataframe(path: Path) -> pd.DataFrame:
    install_pickle_compat()
    with path.open("rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame")
    required = {"timestamp", "throughput"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df.copy()


def split_sessions(df: pd.DataFrame, max_gap_s: float, min_len: int) -> list[pd.DataFrame]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["throughput"] = pd.to_numeric(df["throughput"], errors="coerce")
    df = df.dropna(subset=["timestamp", "throughput"])
    df = df[df["throughput"] > 0].sort_values("timestamp")

    # Collapse duplicate timestamps so the simulator sees a monotonic clock.
    df = df.groupby("timestamp", as_index=False).agg({"throughput": "mean"})
    gaps = df["timestamp"].diff().dt.total_seconds().fillna(0)
    session_id = (gaps > max_gap_s).cumsum()

    sessions: list[pd.DataFrame] = []
    for _, part in df.groupby(session_id):
        if len(part) >= min_len:
            sessions.append(part.reset_index(drop=True))
    return sessions


def assign_split(idx: int, total: int, train_ratio: float, calib_ratio: float) -> str:
    if total == 1:
        return "train"
    train_end = max(1, int(round(total * train_ratio)))
    calib_end = max(train_end + 1, int(round(total * (train_ratio + calib_ratio))))
    if idx < train_end:
        return "train"
    if idx < min(calib_end, total):
        return "calib"
    return "test"


def write_trace(session: pd.DataFrame, out_path: Path) -> None:
    rel_t = (session["timestamp"] - session["timestamp"].iloc[0]).dt.total_seconds()
    bw = session["throughput"].astype(float).clip(lower=0.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([rel_t.to_numpy(dtype=float), bw.to_numpy(dtype=float)])
    np.savetxt(out_path, arr, fmt="%.6f %.6f")


def convert_location(
    loc_key: str,
    source_root: Path,
    output_root: Path,
    max_gap_s: float,
    min_len: int,
    train_ratio: float,
    calib_ratio: float,
) -> dict[str, int]:
    pkl_path = source_root / loc_key / "dataset_tp_sat.pkl"
    df = load_dataframe(pkl_path)
    sessions = split_sessions(df, max_gap_s=max_gap_s, min_len=min_len)

    counts = {"train": 0, "calib": 0, "test": 0}
    for idx, session in enumerate(sessions):
        split = assign_split(idx, len(sessions), train_ratio, calib_ratio)
        counts[split] += 1
        trace_name = f"{loc_key}_session_{idx:03d}.trace"
        write_trace(session, output_root / loc_key / split / trace_name)
    return counts


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=repo_root / "data" / "starnet_pkl")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "safesabr" / "video_trace" / "trace" / "starlink",
    )
    parser.add_argument("--max-gap-s", type=float, default=2.5)
    parser.add_argument("--min-len", type=int, default=120)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--calib-ratio", type=float, default=0.15)
    args = parser.parse_args()

    total_counts: dict[str, dict[str, int]] = {}
    for loc_key in LOCATIONS:
        counts = convert_location(
            loc_key,
            args.source_root,
            args.output_root,
            args.max_gap_s,
            args.min_len,
            args.train_ratio,
            args.calib_ratio,
        )
        total_counts[loc_key] = counts

    print("Converted Starlink traces:")
    for loc_key, counts in total_counts.items():
        print(f"  {LOCATIONS[loc_key]}: {counts}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
