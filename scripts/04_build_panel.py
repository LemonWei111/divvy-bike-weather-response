import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT, ensure_dirs, get_logger, load_config, save_table, write_json


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger("04_build_panel", cfg)

    trips = pd.read_parquet(ROOT / cfg["paths"]["interim"] / "trips_clean.parquet")
    weather = pd.read_parquet(ROOT / cfg["paths"]["interim"] / "weather_daily.parquet")

    trips["date"] = pd.to_datetime(trips["date"])

    dep = (
        trips.dropna(subset=["start_station_id"])
        .groupby(["start_station_id", "date"], as_index=False)
        .agg(departures=("ride_id", "size"),
            member_share=("member_casual", lambda s: (s == "member").mean()))
        .rename(columns={"start_station_id": "station_id"})
    )

    arr = (
        trips.dropna(subset=["end_station_id"])
        .groupby(["end_station_id", "date"], as_index=False)
        .size().rename(columns={"end_station_id": "station_id", "size": "arrivals"})
    )

    panel = dep.merge(arr, on=["station_id", "date"], how="outer")
    panel["departures"] = panel["departures"].fillna(0)
    panel["arrivals"] = panel["arrivals"].fillna(0)
    panel["net_inflow"] = panel["arrivals"] - panel["departures"]

    weather_cols = ["date", "PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND"]
    weather_cols = [c for c in weather_cols if c in weather.columns]
    panel = panel.merge(weather[weather_cols], on="date", how="left")

    stats = {
        "trips_total": len(trips),
        "trips_with_start_station": int(trips["start_station_id"].notna().sum()),
        "trips_with_end_station": int(trips["end_station_id"].notna().sum()),
        "start_coverage": float(trips["start_station_id"].notna().mean()),
        "end_coverage": float(trips["end_station_id"].notna().mean()),
        "panel_rows": len(panel),
        "n_stations": int(panel["station_id"].nunique()),
        "n_days": int(panel["date"].nunique()),
    }
    
    save_table(panel, f"{cfg['paths']['processed']}/station_day_panel.parquet")
    write_json(stats, f"{cfg['paths']['tables']}/panel_stats.json")

    log.info("面板行数: %d，站点数: %d，天数: %d",
             len(panel), panel["station_id"].nunique(), panel["date"].nunique())
    log.info("天气缺失统计:\n%s", panel[weather_cols].isna().sum().to_string())


if __name__ == "__main__":
    main()