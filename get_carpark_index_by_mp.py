# -*- coding: utf-8 -*-
# encoding = utf-8

'''
get_carpark_index_by_mp.py
author：alvin
create dayno: 20210801

Function: Get the car parks' indexes by movingpandas library.
功能: 通过movingpandas获取停车场的索引位置。

History:
version       contributor       comment
v1.0          alvin             第一版
'''



import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import plotly.express as px
from shapely.geometry import Point
from scipy import spatial
import geopandas as gpd
import optuna
import warnings
warnings.filterwarnings("ignore")
import geopandas as gpd
import movingpandas as mpd
from pyproj import CRS
from datetime import datetime, timedelta



# Loading the dataset. 导入数据。
data_dir = Path("../data")
trn_df = pd.read_csv(data_dir / "baseline_locations_train.csv")
tst_df = pd.read_csv(data_dir / "baseline_locations_test.csv")
sub_df = pd.read_csv(data_dir / 'sample_submission.csv')
gt_df = pd.DataFrame()
for (collection_name, phone_name), df in trn_df.groupby(["collectionName", "phoneName"]):
    path = data_dir / f"train/{collection_name}/{phone_name}/ground_truth.csv"
    df = pd.read_csv(path)
    gt_df = pd.concat([gt_df, df]).reset_index(drop=True)
gt_df['phone'] = gt_df['collectionName'] + '_' + gt_df['phoneName']



def get_traj_collection(data):
    '''Transforming the dataset to the Trajectory Collection object throught movingpandas. 
    将数据集转为movingpandas的Trajectory Collection对象。'''
    # millisSinceGpsEpoch -> the formatted time. millisSinceGpsEpoch转换为格式化的日期和时间。
    dt_offset = pd.to_datetime('1980-01-06 00:00:00')
    dt_offset_in_ms = int(dt_offset.value / 1e6)
    data['t'] = pd.to_datetime(data['millisSinceGpsEpoch'] + dt_offset_in_ms, unit='ms')
    data['t'] = pd.to_datetime(data['t'], format='%Y-%m-%d %H:%M:%S')

    # LatDeg & LngDeg -> Point object. 将经纬度转换为Point对象。
    data['geometry'] = [Point(long, lat) for long, lat in zip(data['lngDeg'].to_list(), data['latDeg'].to_list())]
    # Create Geodataframe. 创建Geodataframe. 注意这是: CRS 4326 WGS84。
    geodata = gpd.GeoDataFrame(data, crs = CRS.from_epsg('4326'))
    # Set timestamp as index. 将时间戳设为索引。
    geodata = geodata.set_index('t')
    # Create Trajectory Collection object. 使用Movingpandas创建Trajectory Collection对象，以phone作为轨迹id。
    traj_collection = mpd.TrajectoryCollection(geodata, 'phone')
    return traj_collection



def get_stop_traj(traj_col, min_sec, max_dist):
    '''
    For trajectories, detect stopping points. 针对多个轨迹路线，依次做停车检测。
    Input:
        1. traj_col       (TrajectoryCollection): The Trajectory Collection object involving multiple trajectories. 轨迹路线对象（包括多条轨迹）。
        2. min_sec                       (float): The minimum stop duration(s). 最小停车时长(s)，越大越严格要求停车时长要长。
        3. max_dist                      (float): The maximum stop radius(m). 最大停车半径(m)，越小越严格要求停车范围要小。
    Output:
        1. stop_traj_dict                 (dict): The dict for car parks'indexes. 停车场轨迹索引字典。
                                                  {phone:[start_point_max_idxs, end_point_min_idxs]}
    '''
    traj_num = len(traj_col.trajectories)
    print('The number of trajectories: {}'.format(traj_num))
    stop_traj_dict = {}
    start_point_max_idxs = []
    end_point_min_idxs = []
    for i in range(traj_num):
        tgt_traj = traj_col.trajectories[i]
        tgt_stop = mpd.TrajectoryStopDetector(tgt_traj).get_stop_segments(min_duration=timedelta(seconds=min_sec), max_diameter=max_dist)
        print("'{}':[{},{}],".format(tgt_traj.id, len(tgt_stop.trajectories[0].df), len(tgt_traj.df)-len(tgt_stop.trajectories[-1].df)))
        start_point_max_idxs.append(len(tgt_stop.trajectories[0].df))
        end_point_min_idxs.append(len(tgt_traj.df)-len(tgt_stop.trajectories[-1].df))
    return stop_traj_dict



min_sec = 1
max_dist = 33

traj_col_trn = get_traj_collection(trn_df)
stop_traj_dict_trn = get_stop_traj(traj_col_trn, min_sec, max_dist)

traj_col_tst = get_traj_collection(tst_df)
stop_traj_dict_tst = get_stop_traj(traj_col_tst, min_sec, max_dist)

print('Train:', stop_traj_dict_trn)
print('Test:', stop_traj_dict_tst)




# Return: 返回结果如下：
'''
The number of trajectories: 73
'2020-05-14-US-MTV-1_Pixel4':[79,1688],
'2020-05-14-US-MTV-1_Pixel4XLModded':[76,1739],
'2020-05-14-US-MTV-2_Pixel4':[97,1718],
'2020-05-14-US-MTV-2_Pixel4XLModded':[2,575],
'2020-05-21-US-MTV-1_Pixel4':[318,1922],
'2020-05-21-US-MTV-2_Pixel4':[47,1857],
'2020-05-21-US-MTV-2_Pixel4XL':[49,1790],
'2020-05-29-US-MTV-1_Pixel4':[17,1897],
'2020-05-29-US-MTV-1_Pixel4XL':[49,1894],
'2020-05-29-US-MTV-1_Pixel4XLModded':[8,1894],
'2020-05-29-US-MTV-2_Pixel4':[81,1932],
'2020-05-29-US-MTV-2_Pixel4XL':[83,1936],
'2020-06-04-US-MTV-1_Pixel4':[54,1682],
'2020-06-04-US-MTV-1_Pixel4XL':[58,1681],
'2020-06-04-US-MTV-1_Pixel4XLModded':[58,1767],
'2020-06-05-US-MTV-1_Pixel4':[55,1815],
'2020-06-05-US-MTV-1_Pixel4XL':[118,1880],
'2020-06-05-US-MTV-1_Pixel4XLModded':[190,1121],
'2020-06-05-US-MTV-2_Pixel4':[187,1672],
'2020-06-05-US-MTV-2_Pixel4XL':[18,1611],
'2020-06-11-US-MTV-1_Pixel4':[143,1888],
'2020-06-11-US-MTV-1_Pixel4XL':[154,1772],
'2020-07-08-US-MTV-1_Pixel4':[119,2038],
'2020-07-08-US-MTV-1_Pixel4XL':[121,1868],
'2020-07-08-US-MTV-1_Pixel4XLModded':[121,1244],
'2020-07-17-US-MTV-1_Mi8':[134,2038],
'2020-07-17-US-MTV-2_Mi8':[53,1712],
'2020-08-03-US-MTV-1_Mi8':[241,1935],
'2020-08-03-US-MTV-1_Pixel4':[262,1872],
'2020-08-06-US-MTV-2_Mi8':[66,1717],
'2020-08-06-US-MTV-2_Pixel4':[69,1720],
'2020-08-06-US-MTV-2_Pixel4XL':[70,1723],
'2020-09-04-US-SF-1_Mi8':[83,1714],
'2020-09-04-US-SF-1_Pixel4':[9,1743],
'2020-09-04-US-SF-1_Pixel4XL':[85,1722],
'2020-09-04-US-SF-2_Mi8':[62,2479],
'2020-09-04-US-SF-2_Pixel4':[65,2347],
'2020-09-04-US-SF-2_Pixel4XL':[64,1257],
'2021-01-04-US-RWC-1_Pixel4':[63,2010],
'2021-01-04-US-RWC-1_Pixel4Modded':[62,2010],
'2021-01-04-US-RWC-1_Pixel4XL':[62,2039],
'2021-01-04-US-RWC-1_Pixel5':[65,2009],
'2021-01-04-US-RWC-2_Pixel4':[37,1851],
'2021-01-04-US-RWC-2_Pixel4Modded':[34,1849],
'2021-01-04-US-RWC-2_Pixel4XL':[31,1846],
'2021-01-04-US-RWC-2_Pixel5':[39,1862],
'2021-01-05-US-SVL-1_Mi8':[58,1323],
'2021-01-05-US-SVL-1_Pixel4':[57,1318],
'2021-01-05-US-SVL-1_Pixel4XL':[44,1346],
'2021-01-05-US-SVL-1_Pixel5':[50,1424],
'2021-01-05-US-SVL-2_Pixel4':[45,1146],
'2021-01-05-US-SVL-2_Pixel4Modded':[40,1240],
'2021-01-05-US-SVL-2_Pixel4XL':[17,1135],
'2021-03-10-US-SVL-1_Pixel4XL':[56,1444],
'2021-03-10-US-SVL-1_SamsungS20Ultra':[62,1449],
'2021-04-15-US-MTV-1_Pixel4':[36,1685],
'2021-04-15-US-MTV-1_Pixel4Modded':[41,1682],
'2021-04-15-US-MTV-1_Pixel5':[36,1673],
'2021-04-15-US-MTV-1_SamsungS20Ultra':[38,1683],
'2021-04-22-US-SJC-1_Pixel4':[35,2886],
'2021-04-22-US-SJC-1_SamsungS20Ultra':[34,2811],
'2021-04-26-US-SVL-1_Mi8':[74,1033],
'2021-04-26-US-SVL-1_Pixel5':[72,1031],
'2021-04-28-US-MTV-1_Pixel4':[71,1973],
'2021-04-28-US-MTV-1_Pixel5':[56,1971],
'2021-04-28-US-MTV-1_SamsungS20Ultra':[52,1941],
'2021-04-28-US-SJC-1_Pixel4':[48,1984],
'2021-04-28-US-SJC-1_SamsungS20Ultra':[48,2001],
'2021-04-29-US-MTV-1_Pixel4':[96,1599],
'2021-04-29-US-MTV-1_Pixel5':[15,1587],
'2021-04-29-US-MTV-1_SamsungS20Ultra':[94,1584],
'2021-04-29-US-SJC-2_Pixel4':[28,2316],
'2021-04-29-US-SJC-2_SamsungS20Ultra':[33,2312],

The number of trajectories: 48
'2020-05-15-US-MTV-1_Pixel4':[25,3482],
'2020-05-15-US-MTV-1_Pixel4XL':[957,3498],
'2020-05-28-US-MTV-1_Pixel4':[238,2093],
'2020-05-28-US-MTV-1_Pixel4XL':[181,2095],
'2020-05-28-US-MTV-2_Pixel4':[3,2282],
'2020-05-28-US-MTV-2_Pixel4XL':[4,2214],
'2020-05-28-US-MTV-2_Pixel4XLModded':[2,1456],
'2020-06-04-US-MTV-2_Pixel4':[38,1651],
'2020-06-04-US-MTV-2_Pixel4XL':[43,1649],
'2020-06-04-US-MTV-2_Pixel4XLModded':[39,1661],
'2020-06-10-US-MTV-1_Pixel4':[97,1625],
'2020-06-10-US-MTV-1_Pixel4XL':[98,1624],
'2020-06-10-US-MTV-1_Pixel4XLModded':[95,1631],
'2020-06-10-US-MTV-2_Pixel4':[81,1779],
'2020-06-10-US-MTV-2_Pixel4XL':[83,1770],
'2020-06-10-US-MTV-2_Pixel4XLModded':[22,1930],
'2020-08-03-US-MTV-2_Mi8':[101,1701],
'2020-08-03-US-MTV-2_Pixel4':[103,1694],
'2020-08-03-US-MTV-2_Pixel4XL':[56,1647],
'2020-08-13-US-MTV-1_Mi8':[84,2195],
'2020-08-13-US-MTV-1_Pixel4':[86,2236],
'2021-03-16-US-MTV-2_Pixel4Modded':[10,2027],
'2021-03-16-US-MTV-2_SamsungS20Ultra':[156,2195],
'2021-03-16-US-RWC-2_Pixel4XL':[25,1948],
'2021-03-16-US-RWC-2_Pixel5':[74,1947],
'2021-03-16-US-RWC-2_SamsungS20Ultra':[69,1932],
'2021-03-25-US-PAO-1_Mi8':[85,1719],
'2021-03-25-US-PAO-1_Pixel4':[99,1723],
'2021-03-25-US-PAO-1_Pixel4Modded':[94,1719],
'2021-03-25-US-PAO-1_Pixel5':[96,1723],
'2021-03-25-US-PAO-1_SamsungS20Ultra':[84,1721],
'2021-04-02-US-SJC-1_Pixel4':[69,2315],
'2021-04-02-US-SJC-1_Pixel5':[72,2323],
'2021-04-08-US-MTV-1_Pixel4':[24,1007],
'2021-04-08-US-MTV-1_Pixel4Modded':[48,1005],
'2021-04-08-US-MTV-1_Pixel5':[49,1148],
'2021-04-08-US-MTV-1_SamsungS20Ultra':[48,1008],
'2021-04-21-US-MTV-1_Pixel4':[65,1420],
'2021-04-21-US-MTV-1_Pixel4Modded':[51,1406],
'2021-04-22-US-SJC-2_SamsungS20Ultra':[23,2293],
'2021-04-26-US-SVL-2_SamsungS20Ultra':[32,2301],
'2021-04-28-US-MTV-2_Pixel4':[28,1727],
'2021-04-28-US-MTV-2_SamsungS20Ultra':[49,1751],
'2021-04-29-US-MTV-2_Pixel4':[18,1679],
'2021-04-29-US-MTV-2_Pixel5':[17,1719],
'2021-04-29-US-MTV-2_SamsungS20Ultra':[126,1682],
'2021-04-29-US-SJC-3_Pixel4':[37,1947],
'2021-04-29-US-SJC-3_SamsungS20Ultra':[36,1952],
'''