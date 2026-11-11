import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT, ensure_dirs, get_logger, load_config


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger("05_eda", cfg)

    trips = pd.read_parquet(ROOT / cfg["paths"]["interim"] / "trips_clean.parquet")
    weather = pd.read_parquet(ROOT / cfg["paths"]["interim"] / "weather_daily.parquet")

    daily = (
        trips.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "trips"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.merge(weather, on="date", how="left")

    tables = ROOT / cfg["paths"]["tables"]
    figures = ROOT / cfg["paths"]["figures"]
    daily.to_csv(tables / "daily_summary.csv", index=False)

    # 图 1：日骑行量时间序列
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(daily["date"], daily["trips"], linewidth=1)
    ax.set_title("Daily Divvy Trips")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trips")
    fig.savefig(figures / "daily_trips.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 2：TMAX vs 日骑行量
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(daily["TMAX"], daily["trips"], s=10, alpha=0.7)
    ax.set_title("TMAX vs Daily Trips")
    ax.set_xlabel("TMAX (C)")
    ax.set_ylabel("Trips")
    fig.savefig(figures / "tmax_trips.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 异常日
    daily["z"] = (daily["trips"] - daily["trips"].mean()) / daily["trips"].std()
    anomalies = daily.loc[daily["z"].abs() > 2].copy()
    anomalies.to_csv(tables / "anomaly_days.csv", index=False)

    # 相关性
    corr_cols = [c for c in ["trips", "PRCP", "TMAX", "TMIN", "AWND"] if c in daily.columns]
    daily[corr_cols].corr().to_csv(tables / "daily_corr.csv")

    log.info("异常日数量: %d", len(anomalies))
    log.info("相关性:\n%s", daily[corr_cols].corr().to_string())


if __name__ == "__main__":
    main()