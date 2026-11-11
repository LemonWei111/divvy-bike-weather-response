import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ROOT, ensure_dirs, get_logger, load_config, save_table


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger("03_fetch_weather", cfg)

    raw_path = ROOT / cfg["data"]["weather_raw"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        url = (
            "https://www.ncei.noaa.gov/access/services/data/v1"
            "?dataset=daily-summaries"
            f"&stations={cfg['data']['weather_station']}"
            f"&startDate={cfg['data']['weather_start']}"
            f"&endDate={cfg['data']['weather_end']}"
            f"&dataTypes={cfg['data']['weather_types']}"
            "&format=csv&units=metric"
        )
        log.info("下载天气: %s", url)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        if r.text[:200].lower().startswith("<html"):
            raise RuntimeError("NOAA 返回 HTML，可能是端点错误，请手动下载")
        raw_path.write_bytes(r.content)
    else:
        log.info("天气文件已存在，跳过下载: %s", raw_path)

    weather = pd.read_csv(raw_path)
    if "DATE" not in weather.columns:
        raise ValueError(f"天气文件缺少 DATE 列: {list(weather.columns)}")

    weather["date"] = pd.to_datetime(weather["DATE"]).dt.normalize()
    keep = ["date", "PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND"]
    keep = [c for c in keep if c in weather.columns]
    weather = weather[keep].copy()

    save_table(weather, f"{cfg['paths']['interim']}/weather_daily.parquet")
    log.info("天气行数: %d，日期范围: %s ~ %s", len(weather), weather.date.min(), weather.date.max())


if __name__ == "__main__":
    main()