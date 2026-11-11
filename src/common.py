from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for p in cfg["paths"].values():
        (ROOT / p).mkdir(parents=True, exist_ok=True)


def get_logger(name: str, cfg: dict) -> logging.Logger:
    log_dir = ROOT / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def save_table(df: pd.DataFrame, rel_path: str) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"不支持的输出格式: {path}")


def normalize_station_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "start_time": "started_at",
        "stop_time": "ended_at",
        "usertype": "member_casual",
        "start_station_latitude": "start_lat",
        "start_station_longitude": "start_lng",
        "end_station_latitude": "end_lat",
        "end_station_longitude": "end_lng",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = [
        "ride_id",
        "started_at",
        "ended_at",
        "start_station_id",
        "end_station_id",
        "start_lat",
        "start_lng",
        "member_casual",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列 {missing}；实际列: {list(df.columns)}")
    return df


def load_raw_trips(cfg: dict) -> pd.DataFrame:
    pattern = cfg["data"]["trip_glob"]
    files = sorted(ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有匹配到文件: {pattern}")

    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d = standardize_columns(d)
        d["_source_file"] = f.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def write_json(obj: dict, rel_path: str) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)