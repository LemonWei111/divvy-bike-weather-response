import pandas as pd, glob

for f in sorted(glob.glob('data/raw/trips/*.csv')):
    df = pd.read_csv(f, usecols=['rideable_type','member_casual',
                                 'start_station_id','end_station_id'])
    print(f"\n{f}")
    tab = (df.assign(start_miss=df['start_station_id'].isna(),
                     end_miss=df['end_station_id'].isna())
             .groupby('rideable_type')[['start_miss','end_miss']]
             .mean().round(4))
    print(tab)
    print("缺失行的车型构成：")
    print(df[df['start_station_id'].isna()]['rideable_type']
          .value_counts(normalize=True).round(3))
    
"""
缺失几乎全部集中在 electric_bike，
classic_bike 和 docked_bike 的缺失率接近 0。

站点 ID 缺失是 Divvy 电单车“站点外锁车”造成的结构性缺失，
不是数据录入错误。
站点级分析应分别基于 start_station_id / end_station_id 非空子集，
并在报告中说明覆盖率约 84%，缺失机制与车型相关，可能带来选择偏差。
"""