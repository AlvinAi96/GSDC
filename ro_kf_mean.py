# -*- coding: utf-8 -*-
# encoding = utf-8

'''
ro_kf_mean.py
author：alvin
create dayno: 20210716

Function: Outlier Rejection +   Kalman Filter + Phone Mean.
功能: 异常值剔除 + 卡尔曼滤波 + 平均路径。

History:
version       contributor       comment
v1.0          alvin             第一版

Reference:
1。 Kalman filter - hyperparameter search with BO(Trinh Quoc Anh): https://www.kaggle.com/tqa236/kalman-filter-hyperparameter-search-with-bo
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
import seaborn as sns
from tqdm.notebook import tqdm
import pathlib
import plotly
import plotly.express as px
from shapely.geometry import Point
from scipy import spatial
import geopandas as gpd
from math import sin, cos, atan2, sqrt
import pyproj
import simdkalman
import warnings
warnings.filterwarnings("ignore")



def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    计算地球上两点之间的距离。
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist



def visualize_trafic(df, center, zoom=14):
    fig = px.scatter_mapbox(df,

                            # Here, plotly gets, (x,y) coordinates
                            lat="latDeg",
                            lon="lngDeg",

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




def make_shifted_matrix(vec):
    '''Define shifted matrix'''
    matrix = []
    size = len(vec)
    for i in range(size):
        row = [0] * i + vec[:size-i]
        matrix.append(row)
    return np.array(matrix)



def make_state_vector(T, size):
    '''Define state vector'''
    vector = [1, 0]
    step = 2
    for i in range(size - 2):
        if i % 2 == 0:
            vector.append(T)
            T *= T / step
            step += 1
        else:
            vector.append(0)
    return vector



def make_noise_vector(noise, size):
    '''Define noise vector'''
    noise_vector = []
    for i in range(size):
        if i > 0 and i % 2 == 0:
            noise *= 0.5
        noise_vector.append(noise)
    return noise_vector



def make_kalman_filter(T, size, noise, obs_noise):
    '''Define kalman filter'''
    vec = make_state_vector(T, size)
    state_transition = make_shifted_matrix(vec)
    process_noise = np.diag(make_noise_vector(noise, size)) + np.ones(size) * 1e-9
    observation_model = np.array([[1] + [0] * (size - 1), [0, 1] + [0] * (size - 2)])
    observation_noise = np.diag([obs_noise] * 2) + np.ones(2) * 1e-9
    kf = simdkalman.KalmanFilter(
            state_transition = state_transition,
            process_noise = process_noise,
            observation_model = observation_model,
            observation_noise = observation_noise)
    return kf



def apply_kf_smoothing(df, kf_):
    unique_paths = df[['collectionName', 'phoneName']].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] == collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()
        data = data.reshape(1, len(data), 2)
        smoothed = kf_.smooth(data)
        df.loc[cond, 'latDeg'] = smoothed.states.mean[0, :, 0]
        df.loc[cond, 'lngDeg'] = smoothed.states.mean[0, :, 1]
    return df



def make_lerp_data(df):
    '''
    Generate interpolated lat,lng values for different phone times in the same collection. 插值。
    '''
    org_columns = df.columns

    # Generate a combination of time x collection x phone and combine it with the original data (generate records to be interpolated)
    time_list = df[['collectionName', 'millisSinceGpsEpoch']].drop_duplicates()
    phone_list =df[['collectionName', 'phoneName']].drop_duplicates()
    tmp = time_list.merge(phone_list, on='collectionName', how='outer')

    lerp_df = tmp.merge(df, on=['collectionName', 'millisSinceGpsEpoch', 'phoneName'], how='left')
    lerp_df['phone'] = lerp_df['collectionName'] + '_' + lerp_df['phoneName']
    lerp_df = lerp_df.sort_values(['phone', 'millisSinceGpsEpoch'])

    # linear interpolation
    lerp_df['latDeg_prev'] = lerp_df['latDeg'].shift(1)
    lerp_df['latDeg_next'] = lerp_df['latDeg'].shift(-1)
    lerp_df['lngDeg_prev'] = lerp_df['lngDeg'].shift(1)
    lerp_df['lngDeg_next'] = lerp_df['lngDeg'].shift(-1)
    lerp_df['phone_prev'] = lerp_df['phone'].shift(1)
    lerp_df['phone_next'] = lerp_df['phone'].shift(-1)
    lerp_df['time_prev'] = lerp_df['millisSinceGpsEpoch'].shift(1)
    lerp_df['time_next'] = lerp_df['millisSinceGpsEpoch'].shift(-1)
    # Leave only records to be interpolated
    lerp_df = lerp_df[(lerp_df['latDeg'].isnull())&(lerp_df['phone']==lerp_df['phone_prev'])&(lerp_df['phone']==lerp_df['phone_next'])].copy()
    # calc lerp
    # 当前经纬度 = 历史经纬度 + v * t增量 = 历史经纬度 + ((未来经纬度-历史经纬度)/(未来时间点-当前时间点) * (当前时间点-历史时间点))
    lerp_df['latDeg'] = lerp_df['latDeg_prev'] + ((lerp_df['latDeg_next'] - lerp_df['latDeg_prev']) * ((lerp_df['millisSinceGpsEpoch'] - lerp_df['time_prev']) / (lerp_df['time_next'] - lerp_df['time_prev'])))
    lerp_df['lngDeg'] = lerp_df['lngDeg_prev'] + ((lerp_df['lngDeg_next'] - lerp_df['lngDeg_prev']) * ((lerp_df['millisSinceGpsEpoch'] - lerp_df['time_prev']) / (lerp_df['time_next'] - lerp_df['time_prev'])))

    # Leave only the data that has a complete set of previous and next data.
    lerp_df = lerp_df[~lerp_df['latDeg'].isnull()]

    return lerp_df[org_columns]



def calc_mean_pred(df, lerp_df):
    '''Make a prediction based on the average of the predictions of phones in the same collection. 平均轨迹。'''
    add_lerp = pd.concat([df, lerp_df])
    mean_pred_result = add_lerp.groupby(['collectionName', 'millisSinceGpsEpoch'])[['latDeg', 'lngDeg']].mean().reset_index()
    mean_pred_df = df[['collectionName', 'phoneName', 'millisSinceGpsEpoch']].copy()
    mean_pred_df = mean_pred_df.merge(mean_pred_result[['collectionName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']], on=['collectionName', 'millisSinceGpsEpoch'], how='left')
    return mean_pred_df



def ro_for_trn(base_train, ro_verbose):
    '''Based on the moving distance, do outlier rejection for the train dataset. 基于平移距离，对训练集剔除异常值'''
    print("Outlier Rejection for Train：")
    base_train['collectionName'] = base_train['phone'].apply(lambda x: x.split('_')[0])
    base_train['phoneName'] = base_train['phone'].apply(lambda x: x.split('_')[1])
    cn2pn_trn_df_tst = base_train[['collectionName', 'phoneName']].drop_duplicates()
    train_ro = add_distance_diff(base_train) # 加入平移距离

    train_ro1 = []
    for cname in cn2pn_trn_df_tst['collectionName'].unique():
        for pname in cn2pn_trn_df_tst[cn2pn_trn_df_tst['collectionName']==cname]['phoneName'].unique():
            # Based on Velocity
            tmp_df = train_ro[(train_ro['collectionName']==cname) & (train_ro['phoneName']==pname)]
            next_95 = tmp_df['vel_next'].mean() + (tmp_df['vel_next'].std() * 2)
            prev_95 = tmp_df['vel_prev'].mean() + (tmp_df['vel_prev'].std() * 2)
            # for SJC
            if cname in ['2021-04-22-US-SJC-1', '2021-04-28-US-SJC-1', '2021-04-29-US-SJC-2']:
                ind = tmp_df[(tmp_df['vel_next'] > 20)&(tmp_df['vel_prev'] > 20)][['vel_prev','vel_next']].index
            else:
                ind = tmp_df[(tmp_df['vel_next'] > next_95)&(tmp_df['vel_prev'] > prev_95)][['vel_prev','vel_next']].index
            # fill nan at outlier point
            tmp_df.loc[ind, ['latDeg', 'lngDeg']] = np.nan
            train_ro1.append(tmp_df)
            if ro_verbose == True:
                print('{:<20} | {:<15} | Outlier Number (velocity): {}/{}={}%'.format(cname, pname, len(ind), len(tmp_df), np.round(len(ind)/len(tmp_df)*100,4)))

    base_train = pd.concat(train_ro1, axis = 0)
    print('Done.\n')
    return base_train



def ro_for_tst(sub, ro_verbose):
    '''Based on the moving distance, do outlier rejection for the test dataset.基于平移距离，对测试集剔除异常值'''
    print("Outlier Rejection for Test：")
    sub['collectionName'] = sub['phone'].apply(lambda x: x.split('_')[0])
    sub['phoneName'] = sub['phone'].apply(lambda x: x.split('_')[1])
    cn2pn_trn_df_tst = sub[['collectionName', 'phoneName']].drop_duplicates()
    test_ro = add_distance_diff(sub)

    test_ro1 = []
    for cname in cn2pn_trn_df_tst['collectionName'].unique():
        for pname in cn2pn_trn_df_tst[cn2pn_trn_df_tst['collectionName']==cname]['phoneName'].unique():
            # Based on Velocity
            tmp_df = test_ro[(test_ro['collectionName']==cname) & (test_ro['phoneName']==pname)]
            next_95 = tmp_df['vel_next'].mean() + (tmp_df['vel_next'].std() * 2)
            prev_95 = tmp_df['vel_prev'].mean() + (tmp_df['vel_prev'].std() * 2)
            # for SJC
            if cname in ['2021-04-22-US-SJC-2', '2021-04-29-US-SJC-3']:
                ind = tmp_df[(tmp_df['vel_next'] > 20)&(tmp_df['vel_prev'] > 20)][['vel_prev','vel_next']].index
            else:
                ind = tmp_df[(tmp_df['vel_next'] > next_95)&(tmp_df['vel_prev'] > prev_95)][['vel_prev','vel_next']].index
            # fill nan at outlier point
            tmp_df.loc[ind, ['latDeg', 'lngDeg']] = np.nan
            test_ro1.append(tmp_df)
            if ro_verbose == True:
                print('{:<20} | {:<15} | Outlier Number (velocity): {}/{}={}%'.format(cname, pname, len(ind), len(tmp_df), np.round(len(ind)/len(tmp_df)*100,4)))

    sub = pd.concat(test_ro1, axis = 0)
    print('Done.\n')
    return sub