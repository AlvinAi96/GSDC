# -*- coding: utf-8 -*-
# encoding = utf-8

'''
position_shift.py
author：alvin
create dayno: 20210717

Function: Position Shift.
功能: 位置偏移。

History:
version       contributor       comment
v1.0          alvin             第一版

Reference:
1. 'GSDC: Position shift'(Wojtek Rosa): https://www.kaggle.com/wrrosa/gsdc-position-shift
'''



import numpy as np
import pandas as pd
import os
from pathlib import Path
import pyproj
from pyproj import Proj, transform
import optuna



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



def compute_dist(oof, gt):
    '''Compute the distance between prediction and ground truth. 计算训练集预测结果(oof)和GT(gt)的分数。'''
    gt['phone'] = gt['collectionName'] + '_' + gt['phoneName']
    oof['phone'] = oof['collectionName'] + '_' + gt['phoneName']
    df = oof.merge(gt, on = ['phone','millisSinceGpsEpoch'])
    dst_oof = calc_haversine(df.latDeg_x, df.lngDeg_x, df.latDeg_y, df.lngDeg_y)
    scores = pd.DataFrame({'phone': df.phone, 'dst': dst_oof})
    scores_grp = scores.groupby('phone')
    d50 = scores_grp.quantile(.50).reset_index()
    d50.columns = ['phone','q50']
    d95 = scores_grp.quantile(.95).reset_index()
    d95.columns = ['phone','q95']
    return (scores_grp.quantile(.50).mean() + scores_grp.quantile(.95).mean())/2, d50.merge(d95)



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



def position_shift(df, a):
    '''Position Shift. 位置偏移。
    dist = sqrt( x_diff**2 + y_diff**2 + z_diff**2 )
    P_t = P_t-1 + P_diff * (1 - alpha / dist)
    
    Note: 
    - alpha: a variable. 控制偏移程度的变量。
    - P_diff/dist: the normalized course vector, the same course but with length 1. 将平移距离归一化。
    - alpha * P_diff / dist: the same course but with length a. 将归一化的平移距离放大alpha倍，作为位置偏移量。
    '''
    d = df.copy()
    d['phone'] = d['collectionName'] + '_' + d['phoneName']

    # Tranform to x/y/z coordinate
    d['heightAboveWgs84EllipsoidM'] = 63.5 # the average heightAboveWgs84EllipsoidM of the train dataset
    d['x'], d['y'], d['z'] = zip(*d.apply(lambda x: WGS84_to_ECEF(x.latDeg, x.lngDeg, x.heightAboveWgs84EllipsoidM), axis=1))
    d.sort_values(['phone', 'millisSinceGpsEpoch'], inplace=True)
    for fi in ['x','y','z']:
        d[[fi+'p']] = d[fi].shift().where(d['phone'].eq(d['phone'].shift()))
        d[[fi+'diff']] = d[fi]-d[fi+'p']
    d[['dist']] = np.sqrt(d['xdiff']**2 + d['ydiff']**2+ d['zdiff']**2)
    for fi in ['x','y','z']:
        d[[fi+'new']] = d[fi+'p'] + d[fi+'diff']*(1-a/d['dist'])
    lng, lat, alt = ECEF_to_WGS84(d['xnew'].values,d['ynew'].values,d['znew'].values)
    lng[np.isnan(lng)] = d.loc[np.isnan(lng), 'lngDeg']
    lat[np.isnan(lat)] = d.loc[np.isnan(lat), 'latDeg']
    d['latDeg'] = lat
    d['lngDeg'] = lng
    return d 

