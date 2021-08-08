# -*- coding: utf-8 -*-
# encoding = utf-8

'''
gf_mean.py
author：alvin
create dayno: 20210719

Function: Gaussian Filter + Phone Mean.
功能: 高斯滤波 + 平均路径。

History:
version       contributor       comment
v1.0          alvin             第一版

Reference:
1. 'Adaptive_gauss+phone_mean'(Petr B): https://www.kaggle.com/bpetrb/adaptive-gauss-phone-mean
'''

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import optuna
import os
from pathlib import Path



def apply_gauss_smoothing(df, params):
    '''Apply Gaussian Filter to smooth the data.'''
    df = df.copy()
    SZ_1 = params['sz_1']
    SZ_2 = params['sz_2']
    SZ_CRIT = params['sz_crit']    
    
    unique_paths = df[['collectionName', 'phoneName']].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] == collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()
                
        lat_g1 = gaussian_filter1d(data[:, 0], np.sqrt(SZ_1))
        lon_g1 = gaussian_filter1d(data[:, 1], np.sqrt(SZ_1))
        lat_g2 = gaussian_filter1d(data[:, 0], np.sqrt(SZ_2))
        lon_g2 = gaussian_filter1d(data[:, 1], np.sqrt(SZ_2))

        lat_dif = data[1:,0] - data[:-1,0]
        lon_dif = data[1:,1] - data[:-1,1]

        lat_crit = np.append(np.abs(gaussian_filter1d(lat_dif, np.sqrt(SZ_CRIT)) / (1e-9 + gaussian_filter1d(np.abs(lat_dif), np.sqrt(SZ_CRIT)))),[0])
        lon_crit = np.append(np.abs(gaussian_filter1d(lon_dif, np.sqrt(SZ_CRIT)) / (1e-9 + gaussian_filter1d(np.abs(lon_dif), np.sqrt(SZ_CRIT)))),[0])           
            
        df.loc[cond, 'latDeg'] = lat_g1 * lat_crit + lat_g2 * (1.0 - lat_crit)
        df.loc[cond, 'lngDeg'] = lon_g1 * lon_crit + lon_g2 * (1.0 - lon_crit)                        
    return df



def mean_with_other_phones(df):
    df = df.copy()
    collections_list = df[['collectionName']].drop_duplicates().to_numpy()
    # Target for each colleciton. 针对每个collection
    for collection in collections_list:
        phone_list = df[df['collectionName'].to_list() == collection][['phoneName']].drop_duplicates().to_numpy()

        phone_data = {}
        corrections = {}
        # Target for each phone. 针对每个phone
        for phone in phone_list:
            # Get the boolean of the none value. collection+phone的bool位置
            cond = np.logical_and(df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()
            phone_data[phone[0]] = df[cond][['millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()
        
        # Choose a phone. 选择一个phone的数据
        for current in phone_data:
            correction = np.ones(phone_data[current].shape, dtype=np.float)
            correction[:,1:] = phone_data[current][:,1:] # Load location info. 只载入经纬度，时间全变为1
            
            # Telephones data don't complitely match by time, so - interpolate.
            for other in phone_data:
                if other == current:
                    continue
                # Use other phone to interpolate. 用其它phone做插值    
                # x: timestamp; y: location
                # x为时间phone_data[other][:,0]，y为经纬度phone_data[other][:,1:]
                loc = interp1d(phone_data[other][:,0], 
                               phone_data[other][:,1:], 
                               axis=0, 
                               kind='linear', 
                               copy=False, 
                               bounds_error=None, 
                               fill_value='extrapolate', 
                               assume_sorted=True)
                # In the same collection, find out which points are the start point and stop point
                # 找到同一个collection，哪个点最早和最晚
                start_idx = 0
                stop_idx = 0
                for idx, val in enumerate(phone_data[current][:,0]):
                    if val < phone_data[other][0,0]:
                        start_idx = idx
                    if val < phone_data[other][-1,0]:
                        stop_idx = idx

                if stop_idx - start_idx > 0:
                    correction[start_idx:stop_idx,0] += 1
                    correction[start_idx:stop_idx,1:] += loc(phone_data[current][start_idx:stop_idx,0])                    
            # Mean the trajectorie of other phones. 现有机子和其它机子做平均
            correction[:,1] /= correction[:,0]
            correction[:,2] /= correction[:,0]         
            corrections[current] = correction.copy()
        
        for phone in phone_list:
            cond = np.logical_and(df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()           
            df.loc[cond, ['latDeg', 'lngDeg']] = corrections[phone[0]][:,1:]            
    return df



def calc_haversine(lat1, lon1, lat2, lon2):
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist



def compute_dist(pred_df, gt_df):
    oof = pred_df.copy()
    gt = gt_df.copy()
    df = oof.merge(gt, on = ['phone','millisSinceGpsEpoch'])
    dst_oof = calc_haversine(df.latDeg_x,df.lngDeg_x, df.latDeg_y, df.lngDeg_y)
    scores = pd.DataFrame({'phone': df.phone,'dst': dst_oof})
    scores_grp = scores.groupby('phone')
    d50 = scores_grp.quantile(.50).reset_index()
    d50.columns = ['phone','q50']
    d95 = scores_grp.quantile(.95).reset_index()
    d95.columns = ['phone','q95']
    return (scores_grp.quantile(.50).mean() + scores_grp.quantile(.95).mean())/2, d50.merge(d95)


 