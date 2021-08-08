# -*- coding: utf-8 -*-
# encoding = utf-8

'''
snap_to_grid.py
author：alvin
create dayno: 20210716

Function: Map Matching / Snap to grid.
功能: 路网匹配。

History:
version       contributor       comment
v1.0          alvin             第一版
v2.0          shao              第二版: 加入 限制速度阈值的数据清洗 代码（limit_vel）
v3.0          alvin             第三版: 加入 限制索引范围的数据清洗 代码
v4.0          shao              第四版：snap加入扩展窗口的代码
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
from math import sin, cos, atan2, sqrt
import pyproj
import warnings
warnings.filterwarnings("ignore")



def df_to_gdf(df):
    '''pd.DataFrame -> gpd.GeoDataFrame。'''
    df["geometry"] = [Point(p) for p in df[["lngDeg", "latDeg"]].to_numpy()]
    gdf = gpd.GeoDataFrame(df, geometry=df["geometry"])
    return gdf



def calc_haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points
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



def WGS84_to_ECEF(lat, lon, alt):
    '''Transform WGS84 coordinate system (World Geodetic System 1984, latitude/longitude/altitude)
     to ECEF coordinate system (Earth-Centered, Earth-Fixed, x/y/z).
    将WGS84坐标系（世界大地测量系统，latitude/longitude/altitude）转化为ECEF坐标系（地心地固坐标系，x/y/z）。'''
    # convert to radians. 转换到弧度。
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a    = 6378137.0
    # f is the flattening factor. 扁率。
    finv = 298.257223563
    f = 1 / finv   
    # e is the eccentricity. 偏心率。
    e2 = 1 - (1 - f) * (1 - f)    
    # N is the radius of curvature in the prime vertical. 卯酉圈的曲率半径。
    N = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))
    x = (N + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (N + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (N * (1 - e2) + alt)        * np.sin(rad_lat)
    return x, y, z



def ECEF_to_WGS84(x,y,z):
    '''Transform ECEF coordinate system (Earth-Centered, Earth-Fixed, x/y/z)
     to WGS84 coordinate system (World Geodetic System 1984, latitude/longitude/altitude).
    将ECEF坐标系（地心地固坐标系，x/y/z）转化为WGS84坐标系（世界大地测量系统，latitude/longitude/altitude）。'''  
    transformer = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},)
    lon, lat, alt = transformer.transform(x,y,z,radians=False)
    return lon, lat, alt



def get_xyz(df):
    df['X'], df['Y'], df['Z'] = zip(*df.apply(lambda x: WGS84_to_ECEF(x.latDeg, x.lngDeg, x.heightAboveWgs84EllipsoidM), axis=1))
    return df



def add_xy(df):
    '''(latDeg, lngDeg) -> (x, y) 放在'xy'字段下。'''
    df['xy'] = [(x, y) for x, y in zip(df['latDeg'], df['lngDeg'])]
    return df



def closest_point(lat, lon, height, tree, tgt_line_points):
    '''
    Based on the given grid, find the nearest point by the KD tree.
    给定某经纬度坐标点和路网候选点，使用KDTree，找到该点的最近路网点(x, y, dist)
    Ref: https://www.timvink.nl/closest-coordinates/
    Input:
        1. lat                          (float): Latitude. 纬度(latDeg)。
        2. lon                          (float): Longitude. 经度(lngDeg)。
        3. height                       (float): heightAboveWgs84EllipsoidM. 大地高。
        4. tree                (spatial.KDTree): KDTree.
        5. tgt_line_points   (gpd.GeoDataFrame): The road grid. 路网点。
    Output:
        1. (lat, lon, dist)             (tuple): The matched nearest points on the grid, and distance. 
                                                 匹配上的最近路网点和距离。'''
    try:
        cartesian_coord = WGS84_to_ECEF(lat, lon, height)
        closest = tree.query([cartesian_coord])
        index = closest[1][0]
        return  (tgt_line_points.latDeg[index],
                  tgt_line_points.lngDeg[index],
                  closest[0][0])
    except:
        return (lat, lon, np.inf)



def closest_point_index(lat, lon, height, tree, tgt_line_points):
    '''
    Based on the given grid, find the candidate nearest point's indexes by the KD tree.
    给定某经纬度坐标点和路网候选点，使用KDTree，找到该点的最近路网点(x, y, dist)的索引
    Ref: https://www.timvink.nl/closest-coordinates/
    Input:
        1. lat                          (float): Latitude. 纬度(latDeg)。
        2. lon                          (float): Longitude. 经度(lngDeg)。
        3. height                       (float): heightAboveWgs84EllipsoidM. 大地高。
        4. tree                (spatial.KDTree): KDTree.
        5. tgt_line_points   (gpd.GeoDataFrame): The road grid. 路网点。
    Output:
        1. indexes                      (array): The candidate nearest point's indexes on the grid. 
                                                 匹配上的k个最近邻点的索引'''
    try:
        cartesian_coord = WGS84_to_ECEF(lat, lon, height)
        closest = tree.query([cartesian_coord], k = 10) # 返回k个最近邻点的结果
        indexes = closest[1][0]
        return indexes
    except:
        return np.inf




def get_closest_grid_points(df, line_points, places, tree):
    '''
    Batch Search the nearest grid points (x_, y_).
    批量找到预测点附近最近的路网点（x_, y_）。
    Input：
        1. df               (pd.DataFrame): A collection of data. 某collection下的定位点数据。
        2. line_points  (gpd.GeoDataFrame): The grid points (lat/lng). 道路网点的数据(lat/lng)。
        3. places                  (array): The grid points (x/y/z). 道路网点的数据(x/y/z)。
        4. tree          (spatial.KDTree) : KDTree. 路网点的KD树。
    Output：
        1. df2              (pd.DataFrame): A collection of data with the nearest point (x_,y_) in the grid.
                                            某collection下的定位数据（附路网最近点x_,y_）。'''
    df = add_xy(df)
    ds = []
    for pn, d in tqdm(df.groupby(['phoneName'])):
        d['matched_point'] = [closest_point(d.latDeg.iloc[i], d.lngDeg.iloc[i], d.heightAboveWgs84EllipsoidM.iloc[i], tree, line_points) for i in range(len(d))]
        d['x_'] = d['matched_point'].apply(lambda x: x[0])
        d['y_'] = d['matched_point'].apply(lambda x: x[1])
        ds.append(d)
    df2 = pd.concat(ds)
    return df2



def get_wrong_match_points(df, line_points, places, tree):
    '''
    If any current candidate point show in the last candidate points, save the current optimal points.
    If not, we treat it as a possible wrong match point, which will fill it with none value in the following process.
    如果当前候选点出现在上一个点的候选点中，则保留，否则记录下来，方便后续填空值。
    Input：
        1. df               (pd.DataFrame): A collection of data. 某collection下的定位点数据。
        2. line_points  (gpd.GeoDataFrame): The grid points (lat/lng). 道路网点的数据(lat/lng)
        3. places                  (array): The grid points (x/y/z). 道路网点的数据(xyz)
        4. tree          (spatial.KDTree) : KDTree. 路网点的KD树
    Output：
        1. wrong_point_idx_d(pd.DataFrame): The index of wrong match point under different collection.
                                            不同collection下，存在匹配错误可能的点的索引们。
                                            
    '''
    df = add_xy(df)
    wrong_point_idx_dict = {}
    for pn, d in tqdm(df.groupby(['phoneName'])):
        last_point_idxs = []
        wrong_point_idxs = []
        for i in range(len(d)):
            matched_indexes = closest_point_index(d.latDeg.iloc[i], d.lngDeg.iloc[i], d.heightAboveWgs84EllipsoidM.iloc[i], tree, line_points)
            if len(matched_indexes) == 0:
                # if there is not matched indexes, then pass. 如果匹配不上则跳过
                pass
            elif i < 1:
                # if it is the first point, then pass. 如果是第1个点则赋值后跳过
                last_point_idxs = matched_indexes
                pass
            else:
                # If any current candidate point show in the last candidate points, save the current optimal points.
                # 如果 当前点的候选点 中不存在一个点会出现在 上一个点的候选点 中，则认作 存在匹配错误的可能，记录它的索引位置
                wrong_point_flag = True
                for matched_idx in matched_indexes:
                    if matched_idx in last_point_idxs:
                        wrong_point_flag = False
                        break
                
                if wrong_point_flag == True:
                    wrong_point_idxs.append(i)

                last_point_idxs = matched_indexes
        wrong_point_idx_dict[pn] = wrong_point_idxs  
    return wrong_point_idx_dict



def fill_nan_to_wrong_points(df, wrong_idx_dict):
    '''Fill the wrong points with the none value. 对存在匹配错误的可能的点，填空值。'''
    new_df = []
    for pn, d in df.groupby(['phoneName']):
        d.reset_index(drop = True, inplace = True)
        wrong_idxs = wrong_idx_dict[pn]
        d.loc[wrong_idxs, 'latDeg'] = np.nan
        d.loc[wrong_idxs, 'lngDeg'] = np.nan
        new_df.append(d)
    new_df = pd.concat(new_df, axis = 0)
    return new_df.reset_index(drop = True)



def get_dist_between_pred_and_nearest(df2):
    '''Get the Haversine distance between the original points (latDeg, lngDeg) and the matched grid points (x_, y_).
    获取“baseline预测点(latDeg, lngDeg)”与“路网匹配上的最近点(x_, y_)”的Haversine距离'''
    tmp_dists = []
    for i in range(len(df2)):
        tmp_dist = calc_haversine(df2['latDeg'].iloc[i],
                                  df2['lngDeg'].iloc[i],
                                  df2['x_'].iloc[i],
                                  df2['y_'].iloc[i])
        tmp_dists.append(tmp_dist)
    df2['dist'] = tmp_dists
    return df2



def snap_to_grid(sub, threshold):
    """
    Snap to grid within a threshold. 限制距离阀值。
    Note：
        1. latDeg, lngDeg               are the predicted points. baseline预测点。
        2. x_, y_                       are the closest grid points. 路网最近点。
        3. _x_, _y_                     are the new predictions after post processing. 阈值限制和预处理后的点。

    Input：
        1. sub          (pd.DataFrame): A collection of data with the nearest points(x_, y_) and its distance. 某collection下的定位数据（附路网最近点x_,y_和与其距离）
        2. threshold       (float/int): The threshold of distance between the original point and the matched grid points. 距离阈值。
                                        If the matched grid points is too far away, we fill the none value to it.
                                        若 路网最近点(x_,y_) 与 预测点(latDeg,lngDeg) 在threshold内，
                                        我就把路网点替换预测点
    Output：
        1. df2          (pd.DataFrame): A collection of data with the postprocessed point (_x_, _y_) in the grid. 
                                        某collection下的定位数据（加入阈值后处理后的点(_x_, _y_)）"""
    sub['_x_'] = sub['latDeg']
    sub['_y_'] = sub['lngDeg']
    # 如果 路网最近点 与 预测点 在threshold内，我就把路网点替换预测点。
    # 目的是将 预测点 拉回 路网点, 而超过threshold的点不好把握归到那个路网点，则置空。
    sub.loc[sub['dist'] < threshold, '_x_'] = sub.loc[sub['dist'] < threshold]['x_'] # 偏离路网不太远则匹配
    sub.loc[sub['dist'] < threshold, '_y_'] = sub.loc[sub['dist'] < threshold]['y_']
    sub.loc[sub['dist'] > threshold, '_x_'] = np.nan # 偏离路网太远则置空
    sub.loc[sub['dist'] > threshold, '_y_'] = np.nan
    return sub.copy()



def limit_vel(sub, threshold):
    '''Apply a velocity threshold to filter out outliers. 应用速度阈值剔除异常值'''
    i = 0
    while i < len(sub):
        # if the next velcotiy is large than the threshold, we start to find the outlier
        # 如果下一时刻的速度超过阈值，记录当前时刻的位置
        if sub['vel_next'].iloc[i] > threshold:
            j = i + 1
            # the velocity between point i nad point k is large than threshold, fill the point k with NAN
            # 从i的下一个时刻遍历查找 与当前i时刻平均速度超过阈值的点，若超过阈值 令j的经纬度置nan
            while True:
                if not np.isnan(sub['_x_'].iloc[j]):
                    dist = calc_haversine(
                        sub['_x_'].iloc[i], sub['_y_'].iloc[i],
                        sub['_x_'].iloc[j], sub['_y_'].iloc[j])
                    duration = (sub['millisSinceGpsEpoch'].iloc[j] - sub['millisSinceGpsEpoch'].iloc[i]) // 1000
                    if (dist / duration) > threshold:
                        sub['_x_'].iloc[j] = np.nan
                        sub['_y_'].iloc[j] = np.nan
                        j += 1
                    else:
                        # if point j is fine, then the point i move on 
                        # 若i时刻与j时刻的平均速度 不超过阈值，就让i为j+1
                        i = j + 1
                        break
                else:  # 已经被snap置Nan的情况
                    j += 1
        else:
            i += 1
    return sub.copy()



def extend_window(sub, window_size, window_thresh):
    """
    We find that the points near the outlier also contains uncertainty. Therefore, we fill them with the None value.
    在异常点旁边的点通常也存在不确定性，所以我们给它们赋予空值。
    Input：
        1. sub            (pd.DataFrame): The data. 数据集。
        2. window_size             (int): The size of a empty window. 扩展空窗口的窗口大小。
        3. window_thresh           (int): The threshold of the window size. 
                                          Only the window size rich the threshold, we will extend the empty window size.
                                          为空的窗口要大于一定的阈值才会执行扩展空窗口的操作
    Output:
        1. sub            (pd.DataFrame): The data. 数据集。                                         
    """
    i = 0
    while i < len(sub):
        if np.isnan(sub['_x_'].iloc[i]) or np.isnan(sub['_y_'].iloc[i]):
            j = i + 1
            while True:
                if np.isnan(sub['_x_'].iloc[j]) or np.isnan(sub['_y_'].iloc[j]):
                    j += 1
                else:
                    j -= 1
                    break
            if j - i >= window_thresh:
                min_idx = max(i - window_size, 0)
                max_idx = min(j + window_size + 1, len(sub))
                sub['_x_'].iloc[min_idx: max_idx] = np.nan
                sub['_y_'].iloc[min_idx: max_idx] = np.nan
                i = max_idx
            else:
                i = j + 1
        else:
            i += 1
    return sub.copy()
             


def visualize_trafic(df, center, zoom=15):
    df_copy = df.copy()
    df_copy['index'] = df_copy.index
    fig = px.scatter_mapbox(df_copy,
                            # Here, plotly gets, (x,y) coordinates
                            lat="latDeg",
                            lon="lngDeg",
                            hover_name='index',
                            #Here, plotly detects color of series
                            color="phoneName",
                            labels="phoneName",

                            zoom=zoom,
                            center=center,
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()



def visualize_collection(df, collection):
    target_df = df[df['collectionName']==collection].copy()
    lat_center = target_df['latDeg'].mean()
    lng_center = target_df['lngDeg'].mean()
    center = {"lat":lat_center, "lon":lng_center}
    visualize_trafic(target_df, center)



# Get the score of the train dataset. 训练集分数评估。
def percentile50(x):
    return np.percentile(x, 50)
def percentile95(x):
    return np.percentile(x, 95)
def get_train_score(df, gt):
    gt = gt.rename(columns={'latDeg':'latDeg_gt', 'lngDeg':'lngDeg_gt'})
    df = df.merge(gt, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='inner')
    # calc_distance_error
    df['err'] = calc_haversine(df['latDeg_gt'], df['lngDeg_gt'], df['latDeg'], df['lngDeg'])
    error_df = pd.DataFrame()
    error_df["phoneName"] =  df.phoneName.unique().tolist()
    error_df["dist_50"] = [np.percentile(df[df.phoneName==ph]["err"],50) for ph in error_df["phoneName"].tolist()]
    error_df["dist_95"] = [np.percentile(df[df.phoneName==ph]["err"],95) for ph in error_df["phoneName"].tolist()]
    error_df["avg_dist_50_95"] = np.mean(np.array(error_df.iloc[:,1:]),axis=1)
    print(error_df)
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2
    score = res['p50_p90_mean'].mean()
    return score



def snap_to_grid_for_trn(train_df, gt_df,
                         tgt_collections = ['2021-04-22-US-SJC-1', '2021-04-28-US-SJC-1', '2021-04-29-US-SJC-2'],
                         grid_collection = '2021-04-22-US-SJC-1',
                         grid_phone = 'Pixel4',
                         dist_threshold = 50,
                         vel_th = 30,
                         window_thresh = 2, 
                         window_size = 3):
    '''Implement Snap-to-grid to the train dataset. 对训练集定位点做路网匹配。
    Input:
        1. train_df           (pd.DataFrame): baseline dataset. basline训练集。
        2. gt_df              (pd.DataFrame): gt dataset. gt训练集。
        3. tgt_collections            (list): The collection want to be snap to grid. 要做路网匹配的路线。
        4. grid_collections            (str): The collection used to be snaped. 要被当作路网的路线。
        5. dist_threshold            (float): The threshold of distance between the original point and the matched grid points. 限制匹配距离的阀值。
        6. vel_th                      (int): The velocity threshold to filter out outliers. 限制平均速度的阈值。
        7. window_size                 (int): The size of a empty window. 扩展空窗口的窗口大小。
        8. window_thresh               (int): The threshold of the empty window size. 被扩展的空窗口的长度阈值。
    Output:
        1. train_df2          (pd.DataFrame): The dataset after Snap-to-grid postprocess. 路网匹配后的训练集。
    '''
    print('Snap to Grid for Train：')
    df = train_df.copy()

    final_df_list = []

    for tgt_collection in df.collectionName.unique():
        tgt_df = df[df["collectionName"]==tgt_collection].reset_index(drop=True)
        tgt_gt_df = gt_df[gt_df["collectionName"]==tgt_collection].reset_index(drop=True)

        # Only snap to grid for the targeted collection. 只针对特定的collections做 路网匹配
        if tgt_collection not in tgt_collections:
            final_df_list.append(tgt_df)
            continue
        else:
            print('Snap {} to grid:'.format(tgt_collection))

        # Get the grid trajectory. 
        line_points = df_to_gdf(gt_df[(gt_df["collectionName"]==grid_collection) & (gt_df["phoneName"]==grid_phone)].reset_index(drop=True))
        line_points2 = get_xyz(line_points)
        places0 = np.array(line_points2[['X','Y','Z']])
        tree0 = spatial.KDTree(places0)
        # Get the wrong points' indexes. 获取不同phone下，有哪些点是可能会搭错线的
        wrong_point_idx_dict = get_wrong_match_points(tgt_df, line_points, places0, tree0)
        # Fill nan to those wrong points. 对“搭错线的点”置空
        tgt_df = fill_nan_to_wrong_points(tgt_df, wrong_point_idx_dict)

        # Get the grid trajectory.
        tgt_line_points = df_to_gdf(gt_df[gt_df["collectionName"]==grid_collection].reset_index(drop=True))
        tgt_line_points2 = get_xyz(tgt_line_points)
        places = np.array(tgt_line_points2[['X','Y','Z']])
        tree = spatial.KDTree(places)

        # Snap to Grid. 捕获最近的路网点
        tgt_df2 = get_closest_grid_points(tgt_df, tgt_line_points, places, tree) # 找到预测点附近最近的路网点（x_, y_)
        tgt_df2 = get_dist_between_pred_and_nearest(tgt_df2) # 获取“最近点”与“baseline点”的距离dist
        tgt_df2 = snap_to_grid(tgt_df2, threshold = dist_threshold) # 限制距离阀值
        tgt_df2 = limit_vel(tgt_df2, threshold=vel_th)
        tgt_df2 = extend_window(tgt_df2, window_size=window_size, window_thresh=window_thresh)

        # Evaluation
        # The below postprocess will product the none value, so we need to interpolate the data otherwise we will fail to evalate them.
        # 由于路网匹配的阈值预处理会产生空值，所以要先简单的插值，不然没法统计误差
        tgt_df2.loc[:,['latDeg', 'lngDeg']] = tgt_df2[['latDeg', 'lngDeg']].interpolate(method = 'linear')
        tgt_df2.loc[:,['x_', 'y_']] = tgt_df2[['x_', 'y_']].interpolate(method = 'linear')
        tgt_df2.loc[:,['_x_', '_y_']] = tgt_df2[['_x_', '_y_']].interpolate(method = 'linear')

        # The error between basline and gt
        tmp0 = tgt_df2.copy()
        print('basline点 与 gt点 之间的误差: ', get_train_score(tmp0, tgt_gt_df))
        print('-'*20)
        # The error between grid and gt
        tmp1 = tgt_df2.copy()
        tmp1.drop(['latDeg','lngDeg'], axis=1, inplace=True)
        tmp1.rename(columns={'x_':'latDeg', 'y_':'lngDeg'}, inplace=True)
        print('路网点 与 gt点 之间的误差: ', get_train_score(tmp1, tgt_gt_df))
        print('-'*20)
        # The error between grid with threshold and gt
        tmp3 = tgt_df2.copy()
        tmp3.drop(['latDeg','lngDeg'], axis=1, inplace=True)
        tmp3.rename(columns={'_x_':'latDeg', '_y_':'lngDeg'}, inplace=True)
        print('阈值预处理后的点 与 gt点 之间的误差: ', get_train_score(tmp3, tgt_gt_df))
        print('-'*20)
        print("")

        tgt_df2.drop(['latDeg', 'lngDeg', 'xy', 'matched_point', 'dist', 'x_', 'y_'], axis=1, inplace=True)
        tgt_df2.rename(columns={'_x_':'latDeg', '_y_':'lngDeg'}, inplace=True)
        final_df_list.append(tgt_df2)

    train_df2 = pd.concat(final_df_list, axis=0).reset_index(drop=True)
    print('Done！\n')
    return train_df2



def snap_to_grid_for_tst(test_df, gt_df,
                         tgt_collections = ['2021-04-22-US-SJC-2', '2021-04-29-US-SJC-3'],
                         grid_collection = '2021-04-22-US-SJC-1',
                         grid_phone = 'Pixel4',
                         dist_threshold = 50, 
                         vel_th = 30,
                         window_thresh = 2, 
                         window_size = 3):
    '''Implement Snap-to-grid to the test dataset. 对测试集定位点做路网匹配。
    Input:
        1. test_df            (pd.DataFrame): Test dataset. 测试集。
        2. gt_df              (pd.DataFrame): gt dataset. gt训练集。
        3. tgt_collections            (list): The collection want to be snap to grid. 要做路网匹配的路线。
        4. grid_collections            (str): The collection used to be snaped. 要被当作路网的路线。
        5. dist_threshold            (float): The threshold of distance between the original point and the matched grid points. 限制匹配距离的阀值。
    Output:
        1. test_df2           (pd.DataFrame): The dataset after Snap-to-grid postprocess. 路网匹配后的测试集。
    '''
    print('Snap to Grid for Test：')
    df = test_df.copy()

    final_test_df_list = []

    for tgt_collection in df.collectionName.unique():
        tgt_df = df[df["collectionName"]==tgt_collection].reset_index(drop=True)

        # Only snap to grid for the targeted collection. 只针对特定的collections做 路网匹配
        if tgt_collection not in tgt_collections:
            final_test_df_list.append(tgt_df)
            continue
        else:
            print('Snap {} to grid:'.format(tgt_collection))

        # Get the grid trajectory. 
        line_points = df_to_gdf(gt_df[(gt_df["collectionName"]==grid_collection) & (gt_df["phoneName"]==grid_phone)].reset_index(drop=True))
        line_points2 = get_xyz(line_points)
        places0 = np.array(line_points2[['X','Y','Z']])
        tree0 = spatial.KDTree(places0)
        # Get the wrong points' indexes. 获取不同phone下，有哪些点是可能会搭错线的
        wrong_point_idx_dict = get_wrong_match_points(tgt_df, line_points, places0, tree0)
        # Fill nan to those wrong points. 对“搭错线的点”置空
        tgt_df = fill_nan_to_wrong_points(tgt_df, wrong_point_idx_dict)

        # Get the grid trajectory.
        tgt_line_points = df_to_gdf(gt_df[gt_df["collectionName"]==grid_collection].reset_index(drop=True))
        tgt_line_points2 = get_xyz(tgt_line_points)
        places = np.array(tgt_line_points2[['X','Y','Z']])
        tree = spatial.KDTree(places)

        # Snap to Grid. 捕获最近的路网点
        tgt_df2 = get_closest_grid_points(tgt_df, tgt_line_points, places, tree) # 找到预测点附近最近的路网点（x_, y_)
        tgt_df2 = get_dist_between_pred_and_nearest(tgt_df2) # 获取“最近点”与“baseline点”的距离dist
        tgt_df2 = snap_to_grid(tgt_df2, threshold = dist_threshold) # 限制距离阀值
        tgt_df2 = limit_vel(tgt_df2, threshold=vel_th)
        tgt_df2 = extend_window(tgt_df2, window_size=window_size, window_thresh=window_thresh)

        # Evaluation
        # The below postprocess will product the none value, so we need to interpolate the data otherwise we will fail to evalate them.
        # 由于路网匹配的阈值预处理会产生空值，所以要先简单的插值
        tgt_df2.loc[:,['latDeg', 'lngDeg']] = tgt_df2[['latDeg', 'lngDeg']].interpolate(method = 'linear')
        tgt_df2.loc[:,['x_', 'y_']] = tgt_df2[['x_', 'y_']].interpolate(method = 'linear')
        tgt_df2.loc[:,['_x_', '_y_']] = tgt_df2[['_x_', '_y_']].interpolate(method = 'linear')

        tgt_df2.drop(['latDeg','lngDeg', 'xy', 'matched_point', 'dist', 'x_', 'y_'], axis=1, inplace=True)
        tgt_df2.rename(columns={'_x_':'latDeg', '_y_':'lngDeg'}, inplace=True)
        final_test_df_list.append(tgt_df2)

    test_df2 = pd.concat(final_test_df_list, axis=0).reset_index(drop=True)
    print('Done！\n')
    return test_df2