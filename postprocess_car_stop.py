# -*- coding: utf-8 -*-
# encoding = utf-8

'''
postprocess_car_stop.py
author：alvin
create dayno: 20210716

Function: Post-process the car stop points.
功能: 停车点预处理。

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
import warnings
warnings.filterwarnings("ignore")



def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    计算经纬度两点之间的距离。
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist



def add_distance_diff(df):
    '''Add the 1st differences of distance, duration and velocity. 增加前后 一阶差分 的 距离、时间和速度。'''
    df['latDeg_prev'] = df['latDeg'].shift(1)
    df['latDeg_next'] = df['latDeg'].shift(-1)
    df['lngDeg_prev'] = df['lngDeg'].shift(1)
    df['lngDeg_next'] = df['lngDeg'].shift(-1)
    df['phone_prev'] = df['phone'].shift(1)
    df['phone_next'] = df['phone'].shift(-1)
    df['time_prev'] = df['millisSinceGpsEpoch'].shift(1)
    df['time_next'] = df['millisSinceGpsEpoch'].shift(-1)
    # distance
    df['dist_prev'] = calc_haversine(df['latDeg'], df['lngDeg'], df['latDeg_prev'], df['lngDeg_prev'])
    df['dist_next'] = calc_haversine(df['latDeg'], df['lngDeg'], df['latDeg_next'], df['lngDeg_next'])
    # duration
    df['du_prev'] = (df['millisSinceGpsEpoch'] - df['time_prev']) / 1000
    df['du_next'] = (df['time_next'] - df['millisSinceGpsEpoch']) / 1000
    # velocity
    df['vel_prev'] = df['dist_prev'] / df['du_prev']
    df['vel_next'] = df['dist_next'] / df['du_next']
    df.loc[df['phone']!=df['phone_prev'], ['latDeg_prev', 'lngDeg_prev', 'dist_prev', 'time_prev', 'du_prev']] = np.nan
    df.loc[df['phone']!=df['phone_next'], ['latDeg_next', 'lngDeg_next', 'dist_next', 'time_next', 'du_next']] = np.nan
    return df



def visual_car_stop(df, stop_dist_th = 3):
    '''Visualize the car stop points (the points with the less than 3 meter distance could be treated as the stopping points). 
    可视化停车点（平移距离<3的看作停车点）'''
    # Add the moving distance. 加入平移距离
    new_df = add_distance_diff(df)

    # Gain the stopping points. 获取停车点
    stop_df = []
    for i in range(len(new_df['phone'].unique())):
        tgt_phone = new_df['phone'].unique()[i]
        tmp_df = new_df[(new_df['phone'] == tgt_phone) & (new_df['dist_next'] < stop_dist_th)]
        stop_df.append(tmp_df)
    stop_df = pd.concat(stop_df, axis = 0)
    print('''停车点数：''', len(stop_df))

    # Visualize the car stop points. 可视化停车点
    fig = px.scatter_mapbox(stop_df,

                        # Here, plotly gets, (x,y) coordinates
                        lat="latDeg",
                        lon="lngDeg",
                        text='phoneName',

                        #Here, plotly detects color of series
                        color="phone",
                        labels="phone",

                        zoom=9,
                        center={"lat":37.423576, "lon":-122.094132},
                        height=600,
                        width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()



# The car parks' indexes of the test dataset.
# Highway            : car parks are detected by movingpandas (i.e., postprocess_car_stop.py).
# Street and Downtown: car parks are detected by LGBM (i.e., postprocess_car_lgb.py).
# 
# Note: 
# format: {'phone':[start_point_max_idx, end_point_min_idx]}
# start_point_max_idx: the range [0, start_point_max_idx] is the index range of the begining car park where the car starts to drive.
# end_point_min_idx  : the range [end_point_min_idx, -1] is the index range of the final car park where the car finishes to drive.
# 
# Q：Why we use different postprocess for different types of roads?
# A: We found that highway is much easy to detected the stopping points throught the car moving distance because highway usually doesn't 
# affected by the multipath effect. However, the stop detection in street and downtown is much more difficult. Hence, we use LGB model to
# use the internal IMU data for detecting.  
# 
# 测试集停车场索引边界（highway用movingpandas:min_sec=1,max_dist=33，street和downtown用lgb）
# [出发起始点索引上限start_point_max_idx, 结束终止点索引下限end_point_min_idx]
stop_idx_bound_dict = {
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

'2021-03-16-US-RWC-2_Pixel4XL':[67,1943],
'2021-03-16-US-RWC-2_Pixel5':[63,2002],
'2021-03-16-US-RWC-2_SamsungS20Ultra':[16,1937],
'2021-03-25-US-PAO-1_Mi8':[25,1721],
'2021-03-25-US-PAO-1_Pixel4':[9,1725],
'2021-03-25-US-PAO-1_Pixel4Modded':[2,1721],
'2021-03-25-US-PAO-1_Pixel5':[77,1724],
'2021-03-25-US-PAO-1_SamsungS20Ultra':[15,1723],
'2021-04-02-US-SJC-1_Pixel4':[62,2332],
'2021-04-02-US-SJC-1_Pixel5':[65,2342],
'2021-04-08-US-MTV-1_Pixel4':[41,1021],
'2021-04-08-US-MTV-1_Pixel4Modded':[2,1023],
'2021-04-08-US-MTV-1_Pixel5':[40,1150],
'2021-04-08-US-MTV-1_SamsungS20Ultra':[17,1045],
'2021-04-21-US-MTV-1_Pixel4':[60,1422],
'2021-04-21-US-MTV-1_Pixel4Modded':[5,1413],
'2021-04-26-US-SVL-2_SamsungS20Ultra':[2,2302],
'2021-04-28-US-MTV-2_Pixel4':[13,1734],
'2021-04-28-US-MTV-2_SamsungS20Ultra':[32,1780],
'2021-04-29-US-MTV-2_Pixel4':[124,1681],
'2021-04-29-US-MTV-2_Pixel5':[119,1677],
'2021-04-29-US-MTV-2_SamsungS20Ultra':[120,1719],

'2021-04-22-US-SJC-2_SamsungS20Ultra':[23,2296],
'2021-04-29-US-SJC-3_Pixel4':[34,1952],
'2021-04-29-US-SJC-3_SamsungS20Ultra':[30,1953],  
}



def process_car_park_for_tst(test_df, verbose):
    '''Based on detected car parks' indexes, post-process the car park points by computing the median value to replace them.
    基于检测出的停车场索引，处理测试集的停车场定位点(中位数替换)。
    说明:
        1. 出发起始点索引范围：[0, start_point_max_idx)
        2. 结束终止点索引范围: [end_point_min_idx, -1)
        对索引范围内的点统计中位数，然后用中位数替换掉索引范围内所有点。
    '''
    print('Post-process car park points from the test dataset:')
    new_test_df = []
    for tgt_phone in test_df['phone'].unique():
        tmp_df = test_df[test_df['phone'] == tgt_phone]
        start_point_max_idx, end_point_min_idx = stop_idx_bound_dict[tgt_phone] # get the car park index. 获取出发和结束时停车的上下限
        # Compute the median position of the start car park. 取出发停车时的中位数
        start_median_lat, start_median_lng = tmp_df.iloc[:start_point_max_idx,:][['latDeg','lngDeg']].median()
        tmp_df.iloc[:start_point_max_idx]['latDeg'] = start_median_lat
        tmp_df.iloc[:start_point_max_idx]['lngDeg'] = start_median_lng
        #Compute the median position of the final car park. 取结束停车时的中位数
        end_median_lat, end_median_lng = tmp_df.iloc[end_point_min_idx:,:][['latDeg','lngDeg']].median()
        tmp_df.iloc[end_point_min_idx:]['latDeg'] = end_median_lat
        tmp_df.iloc[end_point_min_idx:]['lngDeg'] = end_median_lng
        if verbose == True:
            print('phone:{:<40}(before {:<3} | after {:<4}):  start_loc:({:.2f},{:.2f}),end_loc:({:.2f},{:.2f})'.format(tgt_phone,
                  start_point_max_idx, end_point_min_idx, start_median_lat, start_median_lng, end_median_lat, end_median_lng))
        new_test_df.append(tmp_df)
    test_df = pd.concat(new_test_df, axis = 0)
    print('Done！\n')
    return test_df



def process_car_stop_for_tst(test_df, stop_dist_th = 3, stop_window = 3):
    '''For the test dataset: we can tell whether the point is stopping or not based on the moving distance, 
    if so, we use calculate the median value to replace the points in the stopping duration.
    针对测试集：根据前后平移距离，判断是否为停车状态。对非停车场的定位点判定停车状态后，采用中位数替换法，替换短期停车窗口内的值。
    Input:
        1. test_df        (pd.DataFrame): The test dataset. 测试集。
        2. stop_dist_th          (float): The moving distance threshold to tell the stopping status. 
                                          判断停车状态时所用的平移距离上限。
        3. stop_window             (int): The tolerance of car stopping. 停车窗口大小，容忍索引位置偏差多少，
                                          举例: 若stop_window=3, 两个停车点索引相差2，则将这两个停车点看作同一段停车时间内的点。
    Output:
        1. new_test_df    (pd.DataFrame): The post-processed test dataset. 中位数替换停车点后的测试集。
    '''
    # Add moving distance. 加入平移距离
    test_ro = add_distance_diff(test_df)

    new_test_df = []
    for i in range(len(test_ro['phone'].unique())):
        # Find out the stopping points of the given trajectories. 找到给定线路的停车点（默认移动距离小于3）
        tgt_phone = test_ro['phone'].unique()[i]
        tgt_df = test_ro[test_ro['phone'] == tgt_phone].reset_index(drop = True)
        tmp_df = tgt_df[tgt_df['dist_next'] < stop_dist_th]
        stop_idxs = list(tmp_df.index)

        # Exclude the points from the car park areas. 排除掉起始和结束的停车点
        start_point_max_idx, end_point_min_idx = stop_idx_bound_dict[tgt_phone]
        stop_idxs = [i for i in stop_idxs if (i > start_point_max_idx) and (i < end_point_min_idx)]

        # Replace the points in the stopping window with the median value. 基于停车点窗口进行中位数替换
        idx_dist_th = stop_window # Tolerate how many non-stopping points can be in the stopping window. 容忍索引位置偏差多少
        idx_windows = []
        lat_windows = []
        lng_windows = []
        for i in range(len(stop_idxs) - 1):
            curr_idx = stop_idxs[i]
            next_idx = stop_idxs[i + 1]
            idx_dist = next_idx - curr_idx
            if idx_dist <= idx_dist_th:
                idx_windows.append(curr_idx)
                lat_windows.append(tgt_df['latDeg'].iloc[curr_idx])
                lng_windows.append(tgt_df['lngDeg'].iloc[curr_idx])
            elif (idx_dist > idx_dist_th) and (idx_windows != []):
                # Compute the median value in the stopping window. 对经纬度窗口内做中位数统计
                lat_m = np.median(lat_windows)
                lng_m = np.median(lng_windows)
                # Replacement. 替换索引窗口内的经纬度
                tgt_df.loc[idx_windows,'latDeg'] = lat_m
                tgt_df.loc[idx_windows,'lngDeg'] = lng_m
                # Clean stopping windows. 清空窗口
                idx_windows = []
                lat_windows = []
                lng_windows = []

        # If the final window still has points, repeat the postprocess again. 若最后停车窗口还有数，则做中位数替换
        if idx_windows != []:
            lat_m = np.median(lat_windows)
            lng_m = np.median(lng_windows)
            tgt_df.loc[idx_windows,'latDeg'] = lat_m
            tgt_df.loc[idx_windows,'lngDeg'] = lng_m

        new_test_df.append(tgt_df)

    new_test_df = pd.concat(new_test_df, axis = 0).reset_index(drop = True)
    return new_test_df



def process_car_stop_for_trn(train_df, stop_dist_th = 3, stop_window = 3, verbose = False):
    '''For the train dataset: we can tell whether the point is stopping or not based on the moving distance, 
    if so, we use calculate the median value to replace the points in the stopping duration.
    Note: Different with process_car_stop_for_tst(), we add the postprocess for the carpark.
    针对训练集：根据前后平移距离，判断是否为停车状态。采用中位数替换停车点。
        1. 停车场停车点处理：认为第一个和最后一个停车窗口是起始和结束停车场。
        2. 行驶中停车点处理：同测试集处理方法。
    Input:
        1. train_df       (pd.DataFrame): The train dataset. 训练集
        2. stop_dist_th          (float): The moving distance threshold to tell the stopping status. 
                                          判断停车状态时所用的平移距离上限
        3. stop_window             (int): The tolerance of car stopping. 停车窗口大小，容忍索引位置偏差多少，
                                          举例: 若stop_window=3, 两个停车点索引相差2，则将这两个停车点看作同一段停车时间内的点
    Output:
        1. new_train_df   (pd.DataFrame): The post-processed train dataset. 中位数替换停车点后的训练集
    '''
    print('Post-process stopping points including the carpark points from the train dataset:')
    # Add moving distance. 加入平移距离
    train_ro = add_distance_diff(train_df)

    train_ro['phone'] = train_ro['collectionName'] + '_' + train_ro['phoneName']
    new_train_df = []
    for i in range(len(train_ro['phone'].unique())):
        # Find out the stopping points of the given trajectories. 找到给定线路的停车点（默认移动距离小于3）
        tgt_phone = train_ro['phone'].unique()[i]
        tgt_df = train_ro[train_ro['phone'] == tgt_phone].reset_index(drop = True)
        tmp_df = tgt_df[tgt_df['dist_next'] < stop_dist_th]
        stop_idxs = list(tmp_df.index)

        # Include the points from the car park areas. 包含起始和结束的停车点
        start_point_max_idx = 0
        end_point_min_idx = len(tgt_df)
        stop_idxs = [i for i in stop_idxs if (i > start_point_max_idx) and (i < end_point_min_idx)]

        #  Replace the points in the stopping window with the median value. 基于停车点窗口进行中位数替换、
        idx_dist_th = stop_window # Tolerate how many non-stopping points can be in the stopping window. 容忍索引位置偏差多少
        idx_windows = []
        lat_windows = []
        lng_windows = []
        window_flag = 1
        start_point_max_idx = 0
        end_point_min_idx = 0
        for i in range(len(stop_idxs) - 1):
            curr_idx = stop_idxs[i]
            next_idx = stop_idxs[i + 1]
            idx_dist = next_idx - curr_idx
            if idx_dist <= idx_dist_th:
                idx_windows.append(curr_idx)
                lat_windows.append(tgt_df['latDeg'].iloc[curr_idx])
                lng_windows.append(tgt_df['lngDeg'].iloc[curr_idx])
            elif (idx_dist > idx_dist_th) and (idx_windows != []):
                # The first stopping window treats as the start carpark area. 若为第一个窗口，则看作起始点停车
                if window_flag == 1:
                    # Compute the median value in the stopping window. 对经纬度窗口内做中位数统计
                    lat_m = np.median(lat_windows)
                    lng_m = np.median(lng_windows)
                    # Replacement. 替换索引窗口内的经纬度
                    tgt_df.loc[:max(idx_windows),'latDeg']  = lat_m
                    tgt_df.loc[:max(idx_windows),'lngDeg'] = lng_m
                    start_point_max_idx = max(idx_windows)
                else:
                    lat_m = np.median(lat_windows)
                    lng_m = np.median(lng_windows)
                    tgt_df.loc[idx_windows,'latDeg']  = lat_m
                    tgt_df.loc[idx_windows,'lngDeg'] = lng_m

                # Clean stopping windows. 清空窗口
                idx_windows = []
                lat_windows = []
                lng_windows = []
                window_flag += 1

        # The final stopping window treats as the final carpark area if the final window still has points.
        # Note: we don't allow the minimum index of the final window is less than the reciprocal 300 indexes.
        # 若最后停车窗口还有数，则做中位数替换(注意这里不允许最后一个窗口的最小停车索引不归属与最后300s内)
        if (idx_windows != []) and (len(tgt_df) - min(idx_windows)) <= 300:
            lat_m = np.median(lat_windows)
            lng_m = np.median(lng_windows)
            tgt_df.loc[min(idx_windows):,'latDeg'] = lat_m
            tgt_df.loc[min(idx_windows):,'lngDeg'] = lng_m
            end_point_min_idx = min(idx_windows)
        else:
            # if the minimum index is too small, we thinks there is not the carpark area.
            # 若最后一个窗口的最小停车索引太靠前了，认做没有结束的停车场定位点
            end_point_min_idx = len(tgt_df)

        new_train_df.append(tgt_df)
        if verbose == True:
            print('phone:{:<40} (before {:<3} | after {:<4}): total_len:{}'.format(tgt_phone, start_point_max_idx, end_point_min_idx, len(tgt_df)))

    new_train_df = pd.concat(new_train_df, axis = 0).reset_index(drop = True)
    print('Done！\n')
    return new_train_df


