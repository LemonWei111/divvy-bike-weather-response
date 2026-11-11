# scripts\02_clean_trips.py
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (
    ensure_dirs,
    get_logger,
    load_config,
    load_raw_trips,
    normalize_station_id,
    save_table,
    write_json,
)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger("02_clean_trips", cfg)

    trips = load_raw_trips(cfg)
    raw_rows = len(trips)
    log.info("原始行数: %d", raw_rows)

    for c in ("started_at", "ended_at"):
        trips[c] = pd.to_datetime(trips[c], errors="coerce")

    trips["duration_min"] = (
        trips["ended_at"] - trips["started_at"]
    ).dt.total_seconds() / 60

    trips["start_station_id"] = normalize_station_id(trips["start_station_id"])
    trips["end_station_id"] = normalize_station_id(trips["end_station_id"])

    trips["member_casual"] = (
        trips["member_casual"]
        .replace({"Subscriber": "member", "Customer": "casual"})
        .astype("string")
        .str.lower()
    )

    c = cfg["clean"]
    mask = (
        trips["duration_min"].between(c["min_duration_min"], c["max_duration_min"])
        & trips["start_lat"].between(c["lat_min"], c["lat_max"])
        & trips["start_lng"].between(c["lng_min"], c["lng_max"])
    )
    clean = trips.loc[mask].copy()
    clean["date"] = clean["started_at"].dt.normalize()

    stats = {
        "raw_rows": raw_rows,
        "clean_rows": len(clean),
        "removed_rows": raw_rows - len(clean),
        "removed_ratio": (raw_rows - len(clean)) / raw_rows if raw_rows else None,
        "min_time": clean["started_at"].min(),
        "max_time": clean["started_at"].max(),
        "n_stations": clean["start_station_id"].nunique(),
        "n_days": clean["date"].nunique(),
    }
    stats["negative_duration_removed"] = int((trips["duration_min"] < 0).sum())
    stats["zero_duration_removed"] = int((trips["duration_min"] == 0).sum())
    stats["sub_minute_removed"] = int(
        trips["duration_min"].between(0, c["min_duration_min"], inclusive="neither").sum()
    )
    stats["over_max_duration_removed"] = int(
        (trips["duration_min"] > c["max_duration_min"]).sum()
    )
    log.info("清洗后行数: %d，删除比例: %.4f", len(clean), stats["removed_ratio"])

    save_table(clean, f"{cfg['paths']['interim']}/trips_clean.parquet")
    write_json(stats, f"{cfg['paths']['tables']}/clean_stats.json")


if __name__ == "__main__":
    main()