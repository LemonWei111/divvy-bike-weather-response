import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT, ensure_dirs, get_logger, load_config


MONTH_COLORS = {
    "2023-05": "tab:blue",
    "2023-06": "tab:orange",
    "2023-07": "tab:green",
    "2023-08": "tab:red",
    "2023-09": "tab:purple",
}


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger("05b_monthly", cfg)

    daily = pd.read_csv(
        ROOT / cfg["paths"]["tables"] / "daily_summary.csv",
        parse_dates=["date"],
    )
    daily = daily.dropna(subset=["trips", "TMAX", "PRCP"]).copy()
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    daily["day_in_month"] = daily["date"].dt.day
    daily["weekday"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["weekday"] >= 5
    daily["rainy"] = daily["PRCP"] > cfg["analysis"]["rain_threshold_mm"]

    # 月内 7 日滚动均值（不跨月）
    daily["trips_smooth"] = np.nan
    for _, g in daily.groupby("month"):
        g = g.sort_values("date")
        daily.loc[g.index, "trips_smooth"] = (
            g["trips"].rolling(7, min_periods=3, center=True).mean().values
        )

    figures = ROOT / cfg["paths"]["figures"]
    tables = ROOT / cfg["paths"]["tables"]

    # 图 1：逐月原始日骑行量叠加
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m, g in daily.groupby("month"):
        ax.plot(g["day_in_month"], g["trips"],
                color=MONTH_COLORS.get(m, "gray"), lw=1, alpha=0.75, label=m)
    ax.set_xlabel("Day of month")
    ax.set_ylabel("Daily trips")
    ax.set_title("Daily Trips by Month (raw, overlaid)")
    ax.legend(fontsize=8, ncol=5)
    fig.savefig(figures / "monthly_overlay_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 2：逐月 7 日滚动均值叠加
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m, g in daily.groupby("month"):
        ax.plot(g["day_in_month"], g["trips_smooth"],
                color=MONTH_COLORS.get(m, "gray"), lw=2, label=m)
    ax.set_xlabel("Day of month")
    ax.set_ylabel("7-day centered mean of trips")
    ax.set_title("Daily Trips by Month (7-day smoothed)")
    ax.legend(fontsize=8, ncol=5)
    fig.savefig(figures / "monthly_overlay_smooth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 3：每月雨天 vs 非雨天日均 trips
    rainy_stats = (
        daily.groupby(["month", "rainy"])["trips"]
        .agg(["mean", "size"])
        .reset_index()
        .pivot(index="month", columns="rainy", values="mean")
        .rename(columns={False: "dry_mean", True: "rainy_mean"})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(rainy_stats))
    w = 0.38
    ax.bar(x - w / 2, rainy_stats["dry_mean"], width=w, label="dry days")
    ax.bar(x + w / 2, rainy_stats["rainy_mean"], width=w, label="rainy days")
    ax.set_xticks(x)
    ax.set_xticklabels(rainy_stats["month"])
    ax.set_ylabel("Mean daily trips")
    ax.set_title("Mean Daily Trips: Rainy vs Dry, by Month")
    ax.legend()
    fig.savefig(figures / "monthly_rain_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 月度统计表
    monthly = (
        daily.groupby("month")
        .agg(
            n_days=("date", "size"),
            trips_mean=("trips", "mean"),
            trips_median=("trips", "median"),
            trips_min=("trips", "min"),
            trips_max=("trips", "max"),
            rainy_days=("rainy", "sum"),
            tmax_mean=("TMAX", "mean"),
            prcp_mean=("PRCP", "mean"),
        )
        .reset_index()
    )
    monthly = monthly.merge(
        rainy_stats[["month", "dry_mean", "rainy_mean"]], on="month", how="left"
    )
    monthly["rain_suppression_pct"] = (
        (monthly["rainy_mean"] - monthly["dry_mean"]) / monthly["dry_mean"] * 100
    ).round(2)
    monthly.round(3).to_csv(tables / "05b_monthly_stats.csv", index=False)

    # 月初/月中/月末分段（仅描述，不推假设）
    daily["month_phase"] = pd.cut(
        daily["day_in_month"], bins=[0, 10, 20, 31],
        labels=["early(1-10)", "mid(11-20)", "late(21-31)"],
        include_lowest=True,
    )
    phase = (
        daily.groupby(["month", "month_phase"], observed=True)["trips"]
        .mean().unstack().round(1)
    )
    phase.to_csv(tables / "05b_month_phase.csv")

    log.info("月度统计:\n%s", monthly.to_string(index=False))
    log.info("月初/月中/月末均值:\n%s", phase.to_string())


if __name__ == "__main__":
    main()