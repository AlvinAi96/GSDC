# -*- coding: utf-8 -*-
# encoding = utf-8

'''
gsdc.py
author：alvin
create dayno: 20210716

Function: Excute all the postprocess.
            1. Position Shift
            2. Stop Detection and Postprocess
            3. Snap to Grid
            4. Gaussian Filter
            5. Kalman Filter
            6. Phone Mean
            7. Position Shift
功能: 汇总后处理过程。
            1. 位置偏移
            2. 停车检测和后处理
            3. 路网匹配
            4. 高斯滤波
            5. 卡尔曼滤波
            6. 平均路径
            7. 位置偏移

History:
version       contributor       comment
v1.0          alvin             第一版
v2.0          alvin             第二版
v3.0          alvin             第三版: 加入movingpandas停车检测代码
v4.0          alvin             第四版: 分路段场景调参优化
v5.0          liu               第五版：对2021-04-22-US-SJC-1 GT数据中某交叉路口的双行道进行平滑,
                                       去掉训练集中downtown路段：2021-04-28–US-SJC1
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

import postprocess_car_stop as pcs
import snap_to_grid as stg
import ro_kf_mean as rkm
import gf_mean as gm
import position_shift as ps



# Load Dataset. 读取数据
data_dir = Path("../data")
trn_df = pd.read_csv(data_dir / "baseline_locations_train.csv")
tst_df = pd.read_csv(data_dir / "baseline_locations_test.csv")
sub_df = pd.read_csv(data_dir / 'sample_submission.csv')
gt_df = pd.DataFrame()
for (collection_name, phone_name), df in pd.read_csv(data_dir / "baseline_locations_train.csv").groupby(["collectionName", "phoneName"]):
    path = data_dir / f"train/{collection_name}/{phone_name}/ground_truth.csv"
    df = pd.read_csv(path)
    gt_df = pd.concat([gt_df, df]).reset_index(drop=True)
gt_df['phone'] = gt_df['collectionName'] + '_' + gt_df['phoneName']


# If you want to repeat again, please open the below comment
# trn_df0 = pd.read_csv(data_dir / "baseline_locations_train.csv")
# trn_df = pd.read_csv("../submit/baseline_locations_train_pcs_stg_gf_kf_mean_shift_sub_0803.csv")
# trn_df = trn_df.merge(trn_df0[['phone', 'millisSinceGpsEpoch', 'heightAboveWgs84EllipsoidM']], on=['phone', 'millisSinceGpsEpoch'], how='left')
# trn_df['collectionName'] = trn_df['phone'].apply(lambda x:x.split('_')[0])
# trn_df['phoneName'] = trn_df['phone'].apply(lambda x:x.split('_')[1])

# tst_df0 = pd.read_csv(data_dir / "baseline_locations_test.csv")
# tst_df = pd.read_csv("../submit/baseline_locations_test_pcs_stg_gf_kf_mean_shift_sub_0803.csv")
# tst_df = tst_df.merge(tst_df0[['phone', 'millisSinceGpsEpoch', 'heightAboveWgs84EllipsoidM']], on=['phone', 'millisSinceGpsEpoch'], how='left')
# tst_df['collectionName'] = tst_df['phone'].apply(lambda x:x.split('_')[0])
# tst_df['phoneName'] = tst_df['phone'].apply(lambda x:x.split('_')[1])



# Visualization
def visualize_trafic(df, center, zoom=10):
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



##############################
# Parameter setting 参数设置
##############################
save_dir = Path("../submit")
save_ver = '0803v2' # 版本号

stop_dist_threshold = 3
stop_window_size = 3
stop_verbose = True

trn_tgt_cols = ['2021-04-22-US-SJC-1', '2021-04-28-US-SJC-1', '2021-04-29-US-SJC-2']
tst_tgt_cols = ['2021-04-22-US-SJC-2', '2021-04-29-US-SJC-3']
grid_cols = '2021-04-22-US-SJC-1'
grid_phone = 'Pixel4'
dist_th = 50
vel_th = 18
window_size = 3
window_thresh = 2
sort_window_size = 20

# Since 2021-04-22-US-SJC-1 is used to be the grid, we don't add it into the train dataset otherwise it will overfit.
# Since 2021-04-28-US-SJC-1 has much more noise, we don't add it into the train dataset.
trn_col_cat_dict = {
'2020-05-14-US-MTV-1':'highway',
'2020-05-14-US-MTV-2':'highway',
'2020-05-21-US-MTV-1':'highway',
'2020-05-21-US-MTV-2':'highway',
'2020-05-29-US-MTV-1':'highway',
'2020-05-29-US-MTV-2':'highway',
'2020-06-04-US-MTV-1':'highway', 
'2020-06-05-US-MTV-1':'highway',
'2020-06-05-US-MTV-2':'highway', 
'2020-06-11-US-MTV-1':'highway',
'2020-07-08-US-MTV-1':'highway', 
'2020-07-17-US-MTV-1':'highway',
'2020-07-17-US-MTV-2':'highway', 
'2020-08-03-US-MTV-1':'highway',
'2020-08-06-US-MTV-2':'highway', 
'2020-09-04-US-SF-1':'highway', 
'2020-09-04-US-SF-2':'highway',
'2021-01-04-US-RWC-1':'highway', 
'2021-01-04-US-RWC-2':'highway',
'2021-01-05-US-SVL-1':'highway', 
'2021-01-05-US-SVL-2':'highway',
'2021-03-10-US-SVL-1':'street', 
'2021-04-15-US-MTV-1':'street',
# '2021-04-22-US-SJC-1':'downtown', 
'2021-04-26-US-SVL-1':'street',
'2021-04-28-US-MTV-1':'street', 
# '2021-04-28-US-SJC-1':'downtown',
'2021-04-29-US-MTV-1':'street', 
'2021-04-29-US-SJC-2':'downtown'
}

tst_col_cat_dict = {
'2020-05-15-US-MTV-1':'highway',
'2020-05-28-US-MTV-1':'highway',
'2020-05-28-US-MTV-2':'highway',
'2020-06-04-US-MTV-2':'highway',
'2020-06-10-US-MTV-1':'highway',
'2020-06-10-US-MTV-2':'highway',
'2020-08-03-US-MTV-2':'highway',
'2020-08-13-US-MTV-1':'highway',
'2021-03-16-US-MTV-2':'highway',
'2021-03-16-US-RWC-2':'street',
'2021-03-25-US-PAO-1':'street',
'2021-04-02-US-SJC-1':'street',
'2021-04-08-US-MTV-1':'street',
'2021-04-21-US-MTV-1':'street',
'2021-04-22-US-SJC-2':'downtown',
'2021-04-26-US-SVL-2':'street',
'2021-04-28-US-MTV-2':'street',
'2021-04-29-US-MTV-2':'street',
'2021-04-29-US-SJC-3':'downtown'    
}


##############################
# 1. Position Shift 位置偏移
##############################
def objective(trial):
    a = trial.suggest_uniform('a', -1, 1)
    trn_shift0 = ps.position_shift(trn_df, a)
    score, scores = ps.compute_dist(trn_shift0, gt_df)
    return score

def find_best_param(n_trials):
    study = optuna.create_study()
    study.optimize(objective, n_trials)
    return study.best_params

best_param = find_best_param(25)
print('The best params (Position Shift): ', best_param['a'])

trn_shift = ps.position_shift(trn_df, best_param['a'])
tst_shift = ps.position_shift(tst_df, best_param['a'])



# Collect the result of different road types. 收集各路段类型的结果
trn_sub = []
tst_sub = []
trn_df_copy = trn_shift.copy()
tst_df_copy = tst_shift.copy()
ground_truth_finetuned = gt_df.copy()
ground_truth = gt_df.copy()


# GT Smoothing Manually. 手动平滑GT路网
# 2021-04-22-US-SJC-1
# pixel4
ground_truth_finetuned['latDeg'].iloc[103937 + 2446] = ground_truth['latDeg'].iloc[103937 + 2340]
ground_truth_finetuned['lngDeg'].iloc[103937 + 2446] = ground_truth['lngDeg'].iloc[103937 + 2340]

ground_truth_finetuned['latDeg'].iloc[103937 + 2445] = ground_truth['latDeg'].iloc[103937 + 2339]
ground_truth_finetuned['lngDeg'].iloc[103937 + 2445] = ground_truth['lngDeg'].iloc[103937 + 2339]

ground_truth_finetuned['latDeg'].iloc[103937 + 2444] = ground_truth['latDeg'].iloc[103937 + 2338]
ground_truth_finetuned['lngDeg'].iloc[103937 + 2444] = ground_truth['lngDeg'].iloc[103937 + 2338]

ground_truth_finetuned['latDeg'].iloc[103937 + 2443] = ground_truth['latDeg'].iloc[103937 + 2333]
ground_truth_finetuned['lngDeg'].iloc[103937 + 2443] = ground_truth['lngDeg'].iloc[103937 + 2333]

ground_truth_finetuned['latDeg'].iloc[103937 + 2442] = ground_truth['latDeg'].iloc[103937 + 2331]
ground_truth_finetuned['lngDeg'].iloc[103937 + 2442] = ground_truth['lngDeg'].iloc[103937 + 2331]

ground_truth_finetuned['latDeg'].iloc[103937 + 2441] = ground_truth['latDeg'].iloc[103937 + 2329]
ground_truth_finetuned['lngDeg'].iloc[103937 + 2441] = ground_truth['lngDeg'].iloc[103937 + 2329]

ground_truth_finetuned['latDeg'].iloc[103937 + 2440] = ground_truth['latDeg'].iloc[103937 + 2322]
ground_truth_finetuned['lngDeg'].iloc[103937 + 2440] = ground_truth['lngDeg'].iloc[103937 + 2322]

ground_truth_finetuned['latDeg'].iloc[103937 + 2439] = ground_truth['latDeg'].iloc[103937 + 2320]
ground_truth_finetuned['lngDeg'].iloc[103937 + 2439] = ground_truth['lngDeg'].iloc[103937 + 2320]

# Samsung
ground_truth_finetuned['latDeg'].iloc[106827 + 2446] = ground_truth['latDeg'].iloc[106827 + 2340]
ground_truth_finetuned['lngDeg'].iloc[106827 + 2446] = ground_truth['lngDeg'].iloc[106827 + 2340]

ground_truth_finetuned['latDeg'].iloc[106827 + 2445] = ground_truth['latDeg'].iloc[106827 + 2339]
ground_truth_finetuned['lngDeg'].iloc[106827 + 2445] = ground_truth['lngDeg'].iloc[106827 + 2339]

ground_truth_finetuned['latDeg'].iloc[106827 + 2444] = ground_truth['latDeg'].iloc[106827 + 2338]
ground_truth_finetuned['lngDeg'].iloc[106827 + 2444] = ground_truth['lngDeg'].iloc[106827 + 2338]

ground_truth_finetuned['latDeg'].iloc[106827 + 2443] = ground_truth['latDeg'].iloc[106827 + 2333]
ground_truth_finetuned['lngDeg'].iloc[106827 + 2443] = ground_truth['lngDeg'].iloc[106827 + 2333]

ground_truth_finetuned['latDeg'].iloc[106827 + 2442] = ground_truth['latDeg'].iloc[106827 + 2331]
ground_truth_finetuned['lngDeg'].iloc[106827 + 2442] = ground_truth['lngDeg'].iloc[106827 + 2331]

ground_truth_finetuned['latDeg'].iloc[106827 + 2441] = ground_truth['latDeg'].iloc[106827 + 2329]
ground_truth_finetuned['lngDeg'].iloc[106827 + 2441] = ground_truth['lngDeg'].iloc[106827 + 2329]

ground_truth_finetuned['latDeg'].iloc[106827 + 2440] = ground_truth['latDeg'].iloc[106827 + 2322]
ground_truth_finetuned['lngDeg'].iloc[106827 + 2440] = ground_truth['lngDeg'].iloc[106827 + 2322]

ground_truth_finetuned['latDeg'].iloc[106827 + 2439] = ground_truth['latDeg'].iloc[106827 + 2320]
ground_truth_finetuned['lngDeg'].iloc[106827 + 2439] = ground_truth['lngDeg'].iloc[106827 + 2320]

gt_df_copy = ground_truth_finetuned.copy()



for col_type in ['highway', 'street', 'downtown']:
    print("Now, let's postprocess the road type：", col_type)
    # Get the collections under the given road type. 收集给定col_type下的collectionName
    tgt_trn_cols = []
    for col_key, col_val in trn_col_cat_dict.items():
        if col_val == col_type:
            tgt_trn_cols.append(col_key)
    
    tgt_tst_cols = []
    for col_key, col_val in tst_col_cat_dict.items():
        if col_val == col_type:
            tgt_tst_cols.append(col_key)

    # Get the dataset under the given road type. 获取特定的数据集
    trn_df = trn_df_copy[trn_df_copy['collectionName'].isin(tgt_trn_cols)]
    tst_df = tst_df_copy[tst_df_copy['collectionName'].isin(tgt_tst_cols)]
    gt_df = gt_df_copy.copy()
    trn_df = trn_df.reset_index(drop = True)
    tst_df = tst_df.reset_index(drop = True)
    gt_df = gt_df.reset_index(drop = True)


    ##############################
    # 2. Stop Detection and Postprocess 停车检测和后处理
    ##############################
    def objective(trial):
        stop_window_size = trial.suggest_int('stop_window_size', 2, 5)
        stop_dist_threshold = trial.suggest_int('stop_dist_threshold', 1, 10)
        tmp_trn_df2 = pcs.process_car_stop_for_trn(trn_df, stop_dist_threshold, stop_window_size, stop_verbose)
        score, scores = gm.compute_dist(tmp_trn_df2, gt_df)
        return score

    def pcs_find_best_param(n_trials):
        study = optuna.create_study()
        study.optimize(objective, n_trials)
        return study.best_params    

    best_param = pcs_find_best_param(10)
    print('The best params (Stop Detection and Postprocess): ', best_param)

    trn_df2 = pcs.process_car_stop_for_trn(trn_df, best_param['stop_dist_threshold'], best_param['stop_window_size'], stop_verbose)
    tst_df1 = pcs.process_car_park_for_tst(tst_df, stop_verbose)
    tst_df2 = pcs.process_car_stop_for_tst(tst_df1, best_param['stop_dist_threshold'], best_param['stop_window_size'])

    trn_df2.to_csv(save_dir / "baseline_locations_train_pcs_{}_{}.csv".format(col_type, save_ver), index=False)
    tst_df2.to_csv(save_dir / "baseline_locations_test_pcs_{}_{}.csv".format(col_type, save_ver), index=False)


    ##############################
    # 3. Snap to Grid 路网匹配
    ##############################
    def objective(trial):
        dist_th = trial.suggest_int('dist_th', 40, 60)
        vel_th = trial.suggest_int('vel_th', 15, 30)
        window_th = trial.suggest_int('window_th', 1, 5)
        window_size = trial.suggest_int('window_size', 1, 5)

        tmp_trn_df3 = stg.snap_to_grid_for_trn(trn_df2, gt_df, trn_tgt_cols, grid_cols, grid_phone, dist_th, vel_th, window_th, window_size)
        score, scores = gm.compute_dist(tmp_trn_df3, gt_df)
        return score   

    def stg_find_best_param(n_trials):
        study = optuna.create_study()
        study.optimize(objective, n_trials)
        return study.best_params    

    best_param = {'dist_th':45, 'vel_th':22, 'window_size':3, 'window_th':2}
    print('The best params (Snap to Grid): ', best_param)

    trn_df3 = stg.snap_to_grid_for_trn(trn_df2, gt_df, trn_tgt_cols, grid_cols, grid_phone, best_param['dist_th'], best_param['vel_th'], best_param['window_th'], best_param['window_size'])
    tst_df3 = stg.snap_to_grid_for_tst(tst_df2, gt_df, tst_tgt_cols, grid_cols, grid_phone, best_param['dist_th'], best_param['vel_th'], best_param['window_th'], best_param['window_size'])

    trn_df3.to_csv(save_dir / "baseline_locations_train_pcs_stg_{}_{}.csv".format(col_type, save_ver), index=False)
    tst_df3.to_csv(save_dir / "baseline_locations_test_pcs_stg_{}_{}.csv".format(col_type, save_ver), index=False)


    ##############################
    # 4-6. Gaussian Filter + Kalman Filter + Phone Mean 高斯滤波 + 卡尔曼滤波 + 平均路径
    ##############################
    trn_ro = trn_df3.copy()
    tst_ro = tst_df3.copy()

    # Gaussian Filter + Kalman Filter. 高斯滤波 + 卡尔曼滤波
    def objective(trial):
        sz_1 = trial.suggest_uniform('sz_1', 0.2, 0.8)
        sz_2 = trial.suggest_uniform('sz_2', 5, 7)
        sz_crit = trial.suggest_uniform('sz_crit', 0.5, 1.5)
        T = trial.suggest_uniform('T', 1, 3)
        size = trial.suggest_int('size', 4, 8)
        noise = trial.suggest_uniform('noise', 1e-06, 3e-06)
        obs_noise = trial.suggest_uniform('obs_noise', 1e-06, 3e-06)    
        # GF
        gf_df = gm.apply_gauss_smoothing(trn_ro, {'sz_1' : sz_1, 'sz_2' :sz_2, 'sz_crit' : sz_crit})
        # KF
        tmp_kf = rkm.make_kalman_filter(T, size, noise, obs_noise)
        trn_gf_kf = rkm.apply_kf_smoothing(gf_df, tmp_kf)
        mean_df = gm.mean_with_other_phones(trn_gf_kf)
        score, scores = gm.compute_dist(mean_df, gt_df)
        return score.values[0]

    def gf_find_best_param(n_trials):
        study = optuna.create_study()
        study.optimize(objective, n_trials)
        return study.best_params

    best_param = gf_find_best_param(30)
    print('The best params (Gaussian + KF + Lerp + Mean): ', best_param)

    trn_gf = gm.apply_gauss_smoothing(trn_ro, {'sz_1':best_param['sz_1'],
                                                'sz_2':best_param['sz_2'],
                                                'sz_crit':best_param['sz_crit']})
    best_kf = rkm.make_kalman_filter(best_param['T'], best_param['size'], best_param['noise'], best_param['obs_noise'])
    trn_gf_kf = rkm.apply_kf_smoothing(trn_gf, best_kf)                                             
    trn_mean = gm.mean_with_other_phones(trn_gf_kf)
    
    tst_gf = gm.apply_gauss_smoothing(tst_ro, {'sz_1':best_param['sz_1'],
                                                'sz_2':best_param['sz_2'],
                                                'sz_crit':best_param['sz_crit']})
    tst_gf_kf = rkm.apply_kf_smoothing(tst_gf, best_kf)                                             
    tst_mean = gm.mean_with_other_phones(tst_gf_kf)                                             

    trn_mean.to_csv(save_dir / "baseline_locations_train_pcs_stg_gf_kf_mean_{}_{}.csv".format(col_type, save_ver), index=False)
    tst_mean.to_csv(save_dir / "baseline_locations_test_pcs_stg_gf_kf_mean_{}_{}.csv".format(col_type, save_ver), index=False)


    ##############################
    # 7. Position Shift 位置偏移
    ##############################
    # Add heightAboveWgs84EllipsoidM. 加入大地高特征
    trn_mean = trn_mean.merge(trn_df[['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'heightAboveWgs84EllipsoidM']], on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='left')
    tst_mean = tst_mean.merge(tst_df[['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'heightAboveWgs84EllipsoidM']], on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='left')

    def objective(trial):
        a = trial.suggest_uniform('a', -1, 1)
        trn_shift = ps.position_shift(trn_mean, a)
        score, scores = ps.compute_dist(trn_shift, gt_df)
        return score

    def find_best_param(n_trials):
        study = optuna.create_study()
        study.optimize(objective, n_trials)
        return study.best_params

    best_param = find_best_param(25)
    print('The best params (Position Shift): ', best_param['a'])

    trn_shift = ps.position_shift(trn_mean, best_param['a'])
    tst_shift = ps.position_shift(tst_mean, best_param['a'])

    # Prepare Submission. 准备提交结果
    trn_shift['phone'] = trn_shift['collectionName'] + '_' + trn_shift['phoneName']
    tst_shift['phone'] = tst_shift['collectionName'] + '_' + tst_shift['phoneName']
    trn_shift = trn_shift[['phone', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']]
    tst_shift = tst_shift[['phone', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']]

    # Save Submission.保存提交结果
    trn_shift.to_csv(save_dir / "baseline_locations_train_pcs_stg_gf_kf_mean_shift_{}_{}.csv".format(col_type, save_ver), index=False)
    tst_shift.to_csv(save_dir / "baseline_locations_test_pcs_stg_gf_kf_mean_shift_{}_{}.csv".format(col_type, save_ver), index=False)

    trn_sub.append(trn_shift)
    tst_sub.append(tst_shift)

    tst_shift['collectionName'] = tst_shift['phone'].apply(lambda x: x.split('_')[0])
    tst_shift['phoneName'] = tst_shift['phone'].apply(lambda x: x.split('_')[1])


# Concatenate the result of different road types. 合并各个路段类型的处理结果
trn_sub = pd.concat(trn_sub, axis = 0)
tst_sub = pd.concat(tst_sub, axis = 0)
trn_sub = trn_sub.reset_index(drop = True)
tst_sub = tst_sub.reset_index(drop = True)

sub_df = sub_df[['phone', 'millisSinceGpsEpoch']].merge(tst_sub, on=['phone', 'millisSinceGpsEpoch'], how='left')
sub_df.drop(['collectionName', 'phoneName'], axis = 1, inplace = True)

trn_sub.to_csv(save_dir / "baseline_locations_train_pcs_stg_gf_kf_mean_shift_sub_{}.csv".format(save_ver), index=False)
sub_df.to_csv(save_dir / "baseline_locations_test_pcs_stg_gf_kf_mean_shift_sub_{}.csv".format(save_ver), index=False)

# Print score
score, scores = gm.compute_dist(trn_sub, gt_df_copy)
print("Score：", score)
print(scores)