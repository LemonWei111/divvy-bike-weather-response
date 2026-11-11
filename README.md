# Divvy + NOAA Weather Exploratory Study

## 数据来源
- Divvy Trips: https://divvy-tripdata.s3.amazonaws.com/
- 下载日期：2026-11-07
- 时间范围：2023-05-01 ~ 2023-09-30
- NOAA Daily Summaries：station USW00094846（Chicago O'Hare）
- 许可说明：Divvy 原始数据不再随仓库分发，代码默认您已经将数据下载到./data/raw/trips目录下

## 环境
pip install -r requirements.txt

## 运行
make inspect
make clean
make weather
make panel
make eda
make monthly
make hypothesis

## 输出
- `output/tables`：统计表与回归结果
- `output/figures`：关键图（`tmax_quadratic.png`、`monthly_rain_compare.png`）
- `output/logs`：运行日志
