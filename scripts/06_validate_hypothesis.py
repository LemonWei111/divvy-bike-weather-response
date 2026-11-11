import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, ensure_dirs, get_logger, load_config


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger("06_validate_hypothesis", cfg)

    daily = pd.read_csv(ROOT / cfg["paths"]["tables"] / "daily_summary.csv",
                        parse_dates=["date"])
    daily["log_trips"] = np.log1p(daily["trips"])
    daily["weekday"] = daily["date"].dt.dayofweek
    daily["weekend"] = (daily["weekday"] >= 5).astype(int)
    daily["rainy"] = (daily["PRCP"] > cfg["analysis"]["rain_threshold_mm"]).astype(int)
    daily = daily.dropna(subset=["trips", "TMAX", "PRCP", "AWND"]).copy()

    # 主模型：温度二次 + 降雨 + 风 + 星期固定效应
    m1 = smf.ols(
        "log_trips ~ TMAX + I(TMAX**2) + PRCP + AWND + C(weekday)",
        data=daily,
    ).fit()

    b1, b2 = m1.params["TMAX"], m1.params["I(TMAX ** 2)"]
    turning = -b1 / (2 * b2) if b2 != 0 else np.nan

    # 稳健性：工作日/周末分开
    m_wd = smf.ols("log_trips ~ TMAX + I(TMAX**2) + PRCP + AWND",
                   data=daily[daily.weekend == 0]).fit()
    m_we = smf.ols("log_trips ~ TMAX + I(TMAX**2) + PRCP + AWND",
                   data=daily[daily.weekend == 1]).fit()

    tables = ROOT / cfg["paths"]["tables"]
    with open(tables / "06_tmax_quadratic_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== 主模型 ===\n")
        f.write(m1.summary().as_text())
        f.write(f"\n\n拐点温度 = {turning:.2f} C\n")
        f.write("\n=== 工作日 ===\n")
        f.write(m_wd.summary().as_text())
        f.write("\n=== 周末 ===\n")
        f.write(m_we.summary().as_text())

    coef = m1.params.rename("coef").to_frame()
    coef["pvalue"] = m1.pvalues
    coef.to_csv(tables / "06_tmax_quadratic_coef.csv")

    # 图 1：TMAX 分箱均值 + 二次拟合（关键图候选）
    daily["tmax_bin"] = pd.cut(daily["TMAX"], bins=np.arange(0, 40, 5))
    binned = (daily.groupby("tmax_bin", observed=True)
                   .agg(tmax_mid=("TMAX", "mean"),
                        trips_mean=("trips", "mean"),
                        n=("trips", "size"))
                   .reset_index(drop=True))

    xx = np.linspace(daily["TMAX"].min(), daily["TMAX"].max(), 100)
    base = {"PRCP": 0.0, "AWND": daily["AWND"].mean(), "weekday": 2}
    pred = m1.get_prediction(pd.DataFrame({**base, "TMAX": xx})).summary_frame()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(daily["TMAX"], daily["trips"], s=14, alpha=0.35, label="daily")
    ax.errorbar(binned["tmax_mid"], binned["trips_mean"],
                yerr=None, fmt="o-", color="tab:orange", label="5C bin mean")
    ax.plot(xx, np.expm1(pred["mean"]), color="tab:red", lw=2,
            label=f"quadratic fit (turning={turning:.1f}C)")
    ax.axvline(turning, color="gray", ls="--", lw=1)
    ax.set_xlabel("TMAX (C)")
    ax.set_ylabel("Trips")
    ax.set_title("TMAX vs Daily Trips: Quadratic Fit")
    ax.legend(fontsize=8)
    fig.savefig(ROOT / cfg["paths"]["figures"] / "tmax_quadratic.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 2：工作日 vs 周末曲线（关键图候选）
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for flag, label, color in [(0, "weekday", "tab:blue"), (1, "weekend", "tab:green")]:
        sub = daily[daily.weekend == flag]
        xx2 = np.linspace(sub["TMAX"].min(), sub["TMAX"].max(), 80)
        base2 = {"PRCP": 0.0, "AWND": sub["AWND"].mean(), "weekday": 2}
        p2 = m1.get_prediction(pd.DataFrame({**base2, "TMAX": xx2})).summary_frame()
        # 注意：这里用 m1 画的是控制后的形状；分组差异看分组模型系数
        ax.plot(xx2, np.expm1(p2["mean"]), color=color, lw=2, label=label)
    ax.set_xlabel("TMAX (C)")
    ax.set_ylabel("Predicted Trips (PRCP=0)")
    ax.set_title("Predicted Trips by TMAX, Weekday vs Weekend")
    ax.legend()
    fig.savefig(ROOT / cfg["paths"]["figures"] / "tmax_weekend.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("主模型 R2=%.3f, TMAX 系数=%.4f, TMAX^2 系数=%.5f (p=%.4f), 拐点=%.2f C",
             m1.rsquared, b1, b2, m1.pvalues["I(TMAX ** 2)"], turning)
    log.info("分组模型 TMAX^2: weekday p=%.4f, weekend p=%.4f",
             m_wd.pvalues["I(TMAX ** 2)"], m_we.pvalues["I(TMAX ** 2)"])


if __name__ == "__main__":
    main()