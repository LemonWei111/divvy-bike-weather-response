from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT, ensure_dirs, get_logger, load_config

DEFAULT_THRESHOLDS = {
    "max_datetime_parse_fail_rate": 0.001,
    "max_missing_ride_id_rate": 0.0,
    "max_ride_id_duplicate_rate": 0.0,
    "max_negative_duration_rate": 0.0,
    "max_zero_duration_rate": 0.01,
    "max_over_24h_rate": 0.01,
    "max_station_id_missing_rate": 0.05,
    "max_coord_missing_rate": 0.05,
    "max_coord_out_of_bound_rate": 0.01,
    "month_match_warn_rate": 0.95,
    "month_match_fail_rate": 0.80,
}

START_CANDIDATES = ["started_at", "start_time"]
END_CANDIDATES = ["ended_at", "stop_time"]
MEMBER_CANDIDATES = ["member_casual", "usertype"]
EXPECTED_MEMBER = {"member", "casual", "Subscriber", "Customer"}
EXPECTED_RIDEABLE = {"electric_bike", "classic_bike", "docked_bike"}


class Checker:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, category: str, check: str, scope: str,
            value, status: str, detail: str = "") -> None:
        self.rows.append({
            "category": category, "check": check, "scope": scope,
            "value": value, "status": status, "detail": detail,
        })

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def safe_rate(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def high_bad(rate: float, warn: float, fail: float) -> str:
    if rate > fail:
        return "FAIL"
    if rate > warn:
        return "WARN"
    return "PASS"


def low_bad(rate: float, warn: float, fail: float) -> str:
    if rate < fail:
        return "FAIL"
    if rate < warn:
        return "WARN"
    return "PASS"


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def df_to_md(df: pd.DataFrame, max_rows: int | None = None,
             max_width: int = 80) -> str:
    if df is None or len(df) == 0:
        return "_无记录_"
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if pd.isna(v):
                s = ""
            else:
                s = str(v).replace("|", "\\|").replace("\n", " ")
                if len(s) > max_width:
                    s = s[:max_width] + "..."
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def profile_one_file(path: Path, cfg: dict, thresholds: dict,
                     checker: Checker, col_profile: list,
                     enum_rows: list, schema_sets: dict,
                     ride_id_parts: list, impact: dict) -> dict:
    log = get_logger("01_inspect_raw", cfg)
    log.info("检查文件: %s", path.name)

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    n = len(df)

    info = {
        "file": path.name,
        "rows": n,
        "n_columns": df.shape[1],
        "columns": "|".join(df.columns),
    }
    schema_sets[path.name] = set(df.columns)

    if n == 0:
        checker.add("文件", "文件非空", path.name, 0, "FAIL", "空文件")
        return info
    checker.add("文件", "文件非空", path.name, n, "PASS")

    # ---- 时间列 ----
    start_col = first_existing(df.columns, START_CANDIDATES)
    end_col = first_existing(df.columns, END_CANDIDATES)

    if start_col is None:
        checker.add("Schema", "缺少起始时间列", path.name,
                    "|".join(START_CANDIDATES), "FAIL")
    if end_col is None:
        checker.add("Schema", "缺少结束时间列", path.name,
                    "|".join(END_CANDIDATES), "FAIL")

    start_dt = end_dt = None
    if start_col is not None:
        if start_col != "started_at":
            checker.add("Schema", "起始时间列名非标准", path.name,
                        start_col, "WARN", "clean 阶段需 rename 为 started_at")
        start_dt = pd.to_datetime(df[start_col], errors="coerce")
        fail_n = int(start_dt.isna().sum())
        fail_rate = safe_rate(fail_n, n)
        checker.add("时间", "started_at 解析失败", path.name,
                    f"{fail_n} ({fail_rate:.4%})",
                    high_bad(fail_rate, 0.0,
                             thresholds["max_datetime_parse_fail_rate"]))
        if start_dt.notna().any():
            info["start_min"] = start_dt.min()
            info["start_max"] = start_dt.max()

        m = re.search(r"(\d{6})", path.name)
        if m:
            ym = m.group(1)
            match_rate = safe_rate(
                (start_dt.dt.strftime("%Y%m") == ym).sum(), n)
            checker.add(
                "时间", "文件月份与起始时间匹配", path.name,
                f"{match_rate:.2%}",
                low_bad(match_rate,
                        thresholds["month_match_warn_rate"],
                        thresholds["month_match_fail_rate"]),
                f"期望月份 {ym}",
            )

    if end_col is not None:
        if end_col != "ended_at":
            checker.add("Schema", "结束时间列名非标准", path.name,
                        end_col, "WARN", "clean 阶段需 rename 为 ended_at")
        end_dt = pd.to_datetime(df[end_col], errors="coerce")
        fail_n = int(end_dt.isna().sum())
        fail_rate = safe_rate(fail_n, n)
        checker.add("时间", "ended_at 解析失败", path.name,
                    f"{fail_n} ({fail_rate:.4%})",
                    high_bad(fail_rate, 0.0,
                             thresholds["max_datetime_parse_fail_rate"]))
        if end_dt.notna().any():
            info["end_min"] = end_dt.min()
            info["end_max"] = end_dt.max()

    # ---- 时长 ----
    dur = None
    if start_dt is not None and end_dt is not None:
        dur = (end_dt - start_dt).dt.total_seconds() / 60
        neg = int((dur < 0).sum())
        zero = int((dur == 0).sum())
        over24 = int((dur > 24 * 60).sum())
        checker.add("时长", "负时长", path.name,
                    f"{neg} ({safe_rate(neg, n):.4%})",
                    high_bad(safe_rate(neg, n), 0.0001,
                            thresholds["max_negative_duration_rate"]),
                    "clean 阶段将过滤 duration_min <= 0")
        checker.add("时长", "零时长", path.name,
                    f"{zero} ({safe_rate(zero, n):.4%})",
                    high_bad(safe_rate(zero, n), 0.0,
                             thresholds["max_zero_duration_rate"]))
        checker.add("时长", "超过 24 小时", path.name,
                    f"{over24} ({safe_rate(over24, n):.4%})",
                    high_bad(safe_rate(over24, n), 0.0,
                             thresholds["max_over_24h_rate"]))
        info["dur_p50"] = round(float(dur.quantile(0.5)), 2)
        info["dur_p99"] = round(float(dur.quantile(0.99)), 2)
        info["dur_max"] = round(float(dur.max()), 2)

    # ---- ride_id ----
    if "ride_id" in df.columns:
        rid = df["ride_id"]
        miss = int(rid.isna().sum())
        dup = int(rid.duplicated().sum())
        checker.add("主键", "ride_id 缺失", path.name,
                    f"{miss} ({safe_rate(miss, n):.4%})",
                    high_bad(safe_rate(miss, n), 0.0,
                             thresholds["max_missing_ride_id_rate"]))
        checker.add("主键", "ride_id 文件内重复", path.name,
                    f"{dup} ({safe_rate(dup, n):.4%})",
                    high_bad(safe_rate(dup, n), 0.0,
                             thresholds["max_ride_id_duplicate_rate"]))
        lens = rid.dropna().astype(str).str.len()
        if len(lens):
            info["ride_id_len_mode"] = int(lens.mode().iloc[0])
        ride_id_parts.append(df[["ride_id"]].copy())
    else:
        checker.add("Schema", "缺少 ride_id", path.name, "", "FAIL")

    # ---- 站点 ID ----
    for side in ("start", "end"):
        col = f"{side}_station_id"
        miss = df[col].isna()
        overall = safe_rate(int(miss.sum()), n)

        # 非电单车缺失才是真问题
        non_eb = df["rideable_type"].ne("electric_bike")
        miss_non_eb = safe_rate(int((miss & non_eb).sum()), int(non_eb.sum()))

        checker.add("站点", f"{col} 缺失（全部）", path.name,
                    f"{int(miss.sum())} ({overall:.2%})", "INFO",
                    "含电单车 dockless 结构性缺失")
        checker.add("站点", f"{col} 缺失（非电单车）", path.name,
                    f"{int((miss & non_eb).sum())} ({miss_non_eb:.4%})",
                    high_bad(miss_non_eb, 0.0,
                            thresholds["max_station_id_missing_rate"]))
        
    # ---- 坐标 ----
    c = cfg["clean"]
    for side in ("start", "end"):
        lat, lng = f"{side}_lat", f"{side}_lng"
        if lat not in df.columns or lng not in df.columns:
            checker.add("地理", f"缺少 {lat}/{lng}", path.name, "", "FAIL")
            continue
        miss = df[lat].isna() | df[lng].isna()
        miss_rate = safe_rate(int(miss.sum()), n)
        checker.add("地理", f"{side} 坐标缺失", path.name,
                    f"{int(miss.sum())} ({miss_rate:.4%})",
                    high_bad(miss_rate, 0.0,
                             thresholds["max_coord_missing_rate"]))
        oob = (~df[lat].between(c["lat_min"], c["lat_max"])) | \
              (~df[lng].between(c["lng_min"], c["lng_max"]))
        oob_rate = safe_rate(int(oob.sum()), n)
        checker.add("地理", f"{side} 坐标越界", path.name,
                    f"{int(oob.sum())} ({oob_rate:.4%})",
                    high_bad(oob_rate, 0.0,
                             thresholds["max_coord_out_of_bound_rate"]))

    # ---- 枚举 ----
    member_col = first_existing(df.columns, MEMBER_CANDIDATES)
    if member_col is None:
        checker.add("枚举", "缺少 member_casual/usertype", path.name, "", "FAIL")
    else:
        if member_col != "member_casual":
            checker.add("枚举", "会员列名非标准", path.name,
                        member_col, "WARN", "clean 阶段需 rename 为 member_casual")
        vc = df[member_col].value_counts(dropna=False)
        unknown = set(vc.index.dropna().astype(str)) - EXPECTED_MEMBER
        checker.add("枚举", "member_casual 未知取值", path.name,
                    ",".join(sorted(unknown)) if unknown else "无",
                    "WARN" if unknown else "PASS")
        for v, cnt in vc.items():
            enum_rows.append({"file": path.name, "column": member_col,
                              "value": v, "count": int(cnt),
                              "share": round(cnt / n, 6)})

    if "rideable_type" in df.columns:
        vc = df["rideable_type"].value_counts(dropna=False)
        unknown = set(vc.index.dropna().astype(str)) - EXPECTED_RIDEABLE
        checker.add("枚举", "rideable_type 未知取值", path.name,
                    ",".join(sorted(unknown)) if unknown else "无",
                    "WARN" if unknown else "PASS")
        for v, cnt in vc.items():
            enum_rows.append({"file": path.name, "column": "rideable_type",
                              "value": v, "count": int(cnt),
                              "share": round(cnt / n, 6)})

    # ---- 清洗规则影响预估（三口径）----
    if dur is not None and "start_station_id" in df.columns and \
            "end_station_id" in df.columns:
        base = dur.between(c["min_duration_min"], c["max_duration_min"]) & \
               df["start_lat"].between(c["lat_min"], c["lat_max"]) & \
               df["start_lng"].between(c["lng_min"], c["lng_max"])

        start_ok = base & df["start_station_id"].notna()
        end_ok = base & df["end_station_id"].notna()

        impact["total_rows"] += n
        impact["kept_all"] += int(base.sum())
        impact["kept_start"] += int(start_ok.sum())
        impact["kept_end"] += int(end_ok.sum())
        impact["dropped_all"] += int((~base).sum())

    # ---- 列 profile ----
    key_cols = {"started_at", "start_time", "ended_at", "stop_time",
                "ride_id", "member_casual", "usertype", "rideable_type",
                "start_station_id", "end_station_id",
                "start_lat", "start_lng", "end_lat", "end_lng"}
    for col in df.columns:
        s = df[col]
        miss_rate = safe_rate(int(s.isna().sum()), n)
        nunique = int(s.nunique(dropna=True)) if col in key_cols else None
        sample = s.dropna().astype(str).head(3).tolist()
        col_profile.append({
            "file": path.name, "column": col, "dtype": str(s.dtype),
            "missing_rate": round(miss_rate, 6),
            "n_unique": nunique,
            "sample": " | ".join(sample)[:120],
            "is_key": col in key_cols,
        })

    return info


def write_report(path: Path, cfg: dict, inventory: pd.DataFrame,
                 checks: pd.DataFrame, col_profile: pd.DataFrame,
                 enum_df: pd.DataFrame, schema_sets: dict,
                 impact: dict, thresholds: dict) -> None:
    status_counts = checks["status"].value_counts().to_dict()
    n_fail = status_counts.get("FAIL", 0)
    n_warn = status_counts.get("WARN", 0)
    n_pass = status_counts.get("PASS", 0)

    lines: list[str] = []
    lines.append("# 原始数据检查报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 配置文件：`{cfg.get('_root')}/config.yaml`")
    lines.append(f"- 文件数：{len(inventory)}")
    lines.append(f"- 总行数：{int(inventory['rows'].sum())}")
    lines.append("")
    lines.append("## 1. 结论速览")
    lines.append("")
    lines.append(f"- FAIL：**{n_fail}** 项")
    lines.append(f"- WARN：**{n_warn}** 项")
    lines.append(f"- PASS：{n_pass} 项")
    if n_fail > 0:
        lines.append("")
        lines.append("> **存在 FAIL 项，不建议直接进入清洗。**"
                     "请先查看第 4 节并处理。")
    else:
        lines.append("")
        lines.append("> 未发现 FAIL 项，可以进入清洗阶段；"
                     "WARN 项建议在报告中说明处理方式。")
    lines.append("")

    lines.append("## 2. 文件清单")
    lines.append("")
    lines.append(df_to_md(inventory, max_width=60))
    lines.append("")

    lines.append("## 3. Schema 一致性")
    lines.append("")
    files = list(schema_sets.keys())
    all_cols = sorted(set().union(*schema_sets.values())) if schema_sets else []
    schema_rows = []
    for col in all_cols:
        row = {"column": col}
        for f in files:
            row[f] = "Y" if col in schema_sets[f] else "-"
        schema_rows.append(row)
    lines.append(df_to_md(pd.DataFrame(schema_rows), max_width=30))
    if len(set(map(frozenset, schema_sets.values()))) == 1:
        lines.append("")
        lines.append("所有文件列集合一致。")
    else:
        lines.append("")
        lines.append("**列集合不一致**，clean 阶段需按并集处理并报告缺失列。")
    lines.append("")

    lines.append("## 4. 质量检查明细")
    lines.append("")
    order = {"FAIL": 0, "WARN": 1, "PASS": 2, "INFO": 3}
    checks_sorted = checks.copy()
    checks_sorted["_o"] = checks_sorted["status"].map(order).fillna(9)
    checks_sorted = checks_sorted.sort_values(
        ["_o", "category", "scope", "check"]).drop(columns="_o")
    lines.append(df_to_md(checks_sorted, max_width=50))
    lines.append("")

    lines.append("## 5. 关键列 Profile")
    lines.append("")
    key_profile = col_profile[col_profile["is_key"]].copy()
    lines.append(df_to_md(key_profile, max_rows=120, max_width=40))
    lines.append("")
    lines.append("全量列 profile 见 `output/tables/01_column_profile.csv`。")
    lines.append("")

    lines.append("## 6. 枚举取值分布")
    lines.append("")
    lines.append(df_to_md(enum_df, max_rows=120, max_width=40))
    lines.append("")

    lines.append("## 7. 清洗规则影响预估")
    lines.append("")
    total = impact.get("total_rows", 0)
    rows = [
        {"口径": "原始总行数", "行数": total, "占比": "100%"},
        {"口径": "全行保留（时长+起点坐标，不要求站点）",
         "行数": impact["kept_all"],
         "占比": f"{safe_rate(impact['kept_all'], total):.2%}"},
        {"口径": "起点站点非空（用于 departures）",
         "行数": impact["kept_start"],
         "占比": f"{safe_rate(impact['kept_start'], total):.2%}"},
        {"口径": "终点站点非空（用于 arrivals）",
         "行数": impact["kept_end"],
         "占比": f"{safe_rate(impact['kept_end'], total):.2%}"},
        {"口径": "全行口径删除", "行数": impact["dropped_all"],
         "占比": f"{safe_rate(impact['dropped_all'], total):.2%}"},
    ]
    lines.append(df_to_md(pd.DataFrame(rows)))
    lines.append("")
    lines.append("规则来源：`config.yaml` 的 `clean` 段"
                 "（时长、站点缺失、起点坐标边界）。"
                 "组合过滤的实际影响以 `02_clean_trips.py` 输出为准。")
    lines.append("")

    lines.append("## 8. 阈值")
    lines.append("")
    thr_df = pd.DataFrame(
        [{"阈值": k, "值": v} for k, v in thresholds.items()])
    lines.append(df_to_md(thr_df))
    lines.append("")

    lines.append("## 9. 下一步")
    lines.append("")
    lines.append("1. 处理 FAIL 项；若为列名问题，在 clean 阶段 rename。")
    lines.append("2. 对 WARN 项在报告中写明处理策略。")
    lines.append("3. 运行 `make clean`，核对实际删除行数与本节预估是否接近。")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="存在 FAIL 时以非零码退出")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger("01_inspect_raw", cfg)

    thresholds = {**DEFAULT_THRESHOLDS, **cfg.get("inspect", {})}
    files = sorted(ROOT.glob(cfg["data"]["trip_glob"]))
    if not files:
        log.error("没有匹配到文件: %s", cfg["data"]["trip_glob"])
        sys.exit(1)

    checker = Checker()
    col_profile: list = []
    enum_rows: list = []
    schema_sets: dict = {}
    ride_id_parts: list = []
    impact = {"total_rows": 0, "kept_all": 0, "kept_start": 0,
            "kept_end": 0, "dropped_all": 0}
    inventory_rows: list = []

    for f in files:
        info = profile_one_file(f, cfg, thresholds, checker,
                                col_profile, enum_rows, schema_sets,
                                ride_id_parts, impact)
        inventory_rows.append(info)

    # 全局 ride_id 重复
    if ride_id_parts:
        all_rid = pd.concat(ride_id_parts, ignore_index=True)["ride_id"]
        dup = int(all_rid.duplicated().sum())
        checker.add("主键", "ride_id 全局重复", "all_files",
                    f"{dup} ({safe_rate(dup, len(all_rid)):.4%})",
                    high_bad(safe_rate(dup, len(all_rid)), 0.0,
                             thresholds["max_ride_id_duplicate_rate"]))

    inventory = pd.DataFrame(inventory_rows)
    checks = checker.to_df()
    col_df = pd.DataFrame(col_profile)
    enum_df = pd.DataFrame(enum_rows)

    tables = ROOT / cfg["paths"]["tables"]
    reports = ROOT / "output" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(tables / "01_file_inventory.csv", index=False)
    checks.to_csv(tables / "01_quality_checks.csv", index=False)
    col_df.to_csv(tables / "01_column_profile.csv", index=False)
    enum_df.to_csv(tables / "01_enum_values.csv", index=False)

    schema_rows = []
    files_order = list(schema_sets.keys())
    all_cols = sorted(set().union(*schema_sets.values())) if schema_sets else []
    for col in all_cols:
        row = {"column": col}
        for f in files_order:
            row[f] = 1 if col in schema_sets[f] else 0
        schema_rows.append(row)
    pd.DataFrame(schema_rows).to_csv(
        tables / "01_schema_matrix.csv", index=False)

    report_path = reports / "01_raw_inspection.md"
    write_report(report_path, cfg, inventory, checks, col_df,
                 enum_df, schema_sets, impact, thresholds)

    n_fail = int((checks["status"] == "FAIL").sum())
    log.info("报告已写入 %s", report_path)
    log.info("检查项总数: %d, FAIL: %d, WARN: %d",
             len(checks), n_fail, int((checks["status"] == "WARN").sum()))

    if n_fail > 0:
        log.warning("存在 %d 个 FAIL 项，请查看报告第 4 节", n_fail)
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()