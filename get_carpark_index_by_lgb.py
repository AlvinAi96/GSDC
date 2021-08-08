# -*- coding: utf-8 -*-
# encoding = utf-8

'''
get_carpark_index_by_lgb.py
author：alvin
create dayno: 20210801

Function: Get the car parks' indexes by LightGBM model.
功能: 通过LighGBM获取停车场的索引位置。

History:
version       contributor       comment
v1.0          alvin             第一版

Reference:
1. 'A car is Moving or Not?? Accuracy 94%!'(mashrimp): https://www.kaggle.com/katomash/a-car-is-moving-or-not-accuracy-94
'''



import pandas as pd
import pathlib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



def make_gt(path, collectionName, phoneName):
    '''Loading ground truth and baseline file for the train dataset. 读取ground truth和baseline文件进入训练集。'''
    p = pathlib.Path(path)
    gt_files = list(p.glob('train/*/*/ground_truth.csv'))

    gts = []
    for gt_file in gt_files:
        gts.append(pd.read_csv(gt_file))
    ground_truth = pd.concat(gts)
    
    cols = ['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']
    baseline = pd.read_csv(path + '/baseline_locations_train.csv', usecols=cols)
    ground_truth = ground_truth.merge(baseline, how='inner', on=cols[:3], suffixes=('_gt', '_bs'))
    ground_truth["millisSinceGpsEpoch"] = ground_truth["millisSinceGpsEpoch"]//1000
    if (collectionName is None) or (phoneName is None):
        return ground_truth
    else:
        return ground_truth[(ground_truth['collectionName'] == collectionName) & (ground_truth['phoneName'] == phoneName)]
    


def make_gt_tst(path, collectionName, phoneName):
    '''Loading baseline file for the test dataset. 读取baseline文件进入测试集。'''
    cols = ['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']
    baseline = pd.read_csv(path + '/baseline_locations_test.csv', usecols=cols)
    baseline['latDeg_bs'] = baseline['latDeg']
    baseline['lngDeg_bs'] = baseline['lngDeg']
    baseline["millisSinceGpsEpoch"] = baseline["millisSinceGpsEpoch"]//1000
    if (collectionName is None) or (phoneName is None):
        return baseline
    else:
        return baseline[(baseline['collectionName'] == collectionName) & (baseline['phoneName'] == phoneName)]



def make_tag(df, tag_v):
    '''Based on speed, make Car Stop Tag for labeling. 基于速度, 制作训练集的标签。'''
    df.loc[df['speedMps'] < tag_v, 'tag'] = 1
    df.loc[df['speedMps'] >= tag_v, 'tag'] = 0
    return df



def gnss_log_to_dataframes(path):
    '''Loading Gnss Log File. 加载GNSS日志'''
    print('Loading ' + path, flush=True)
    gnss_section_names = {'Raw', 'UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    with open(path) as f_open:
        datalines = f_open.readlines()

    datas = {k: [] for k in gnss_section_names}
    gnss_map = {k: [] for k in gnss_section_names}
    for dataline in datalines:
        is_header = dataline.startswith('#')
        dataline = dataline.strip('#').strip().split(',')
        # skip over notes, version numbers, etc
        if is_header and dataline[0] in gnss_section_names:
            try:
                gnss_map[dataline[0]] = dataline[1:]
            except:
                pass
        elif not is_header:
            try:
                datas[dataline[0]].append(dataline[1:])
            except:
                pass
    results = dict()
    for k, v in datas.items():
        results[k] = pd.DataFrame(v, columns=gnss_map[k])
    # pandas doesn't properly infer types from these lists by default
    for k, df in results.items():
        for col in df.columns:
            if col == 'CodeType':
                continue
            try:
                results[k][col] = pd.to_numeric(results[k][col])
            except:
                pass
    return results



def add_IMU(df, INPUT, cname, pname):
    '''Adding IMU Data into the given dataframe for the train dataset . 对训练集加入IMU数据。'''
    path = INPUT + "/train/"+cname+"/"+pname+"/"+pname+"_GnssLog.txt"
    gnss_dfs = gnss_log_to_dataframes(path)
    acce_df = gnss_dfs["UncalAccel"]
    magn_df = gnss_dfs["UncalMag"]
    gyro_df = gnss_dfs["UncalGyro"]
    
    acce_df["millisSinceGpsEpoch"] = acce_df["utcTimeMillis"] - 315964800000
    acce_df["millisSinceGpsEpoch"] = acce_df["millisSinceGpsEpoch"]//1000 +18
    magn_df["millisSinceGpsEpoch"] = magn_df["utcTimeMillis"] - 315964800000
    magn_df["millisSinceGpsEpoch"] = magn_df["millisSinceGpsEpoch"]//1000 +18
    gyro_df["millisSinceGpsEpoch"] = gyro_df["utcTimeMillis"] - 315964800000
    gyro_df["millisSinceGpsEpoch"] = gyro_df["millisSinceGpsEpoch"]//1000 +18
    
    acce_df["x_f_acce"] = acce_df["UncalAccelZMps2"]
    acce_df["y_f_acce"] = acce_df["UncalAccelXMps2"]
    acce_df["z_f_acce"] = acce_df["UncalAccelYMps2"]
    # magn 
    magn_df["x_f_magn"] = magn_df["UncalMagZMicroT"]
    magn_df["y_f_magn"] = magn_df["UncalMagYMicroT"]
    magn_df["z_f_magn"] = magn_df["UncalMagXMicroT"]
    # gyro
    gyro_df["x_f_gyro"] = gyro_df["UncalGyroXRadPerSec"]
    gyro_df["y_f_gyro"] = gyro_df["UncalGyroYRadPerSec"]
    gyro_df["z_f_gyro"] = gyro_df["UncalGyroZRadPerSec"]    

    df = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps"]].sort_values('millisSinceGpsEpoch'), acce_df[["millisSinceGpsEpoch", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    df = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), magn_df[["millisSinceGpsEpoch", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    df = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps", "x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), gyro_df[["millisSinceGpsEpoch", "x_f_gyro", "y_f_gyro", "z_f_gyro"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    return df



def add_IMU_tst(df, INPUT, cname, pname):
    '''Adding IMU Data into the given dataframe for the test dataset . 对测试集加入IMU数据。'''
    path = INPUT + "/test/"+cname+"/"+pname+"/"+pname+"_GnssLog.txt"
    gnss_dfs = gnss_log_to_dataframes(path)
    acce_df = gnss_dfs["UncalAccel"]
    magn_df = gnss_dfs["UncalMag"]
    gyro_df = gnss_dfs["UncalGyro"]
    
    acce_df["millisSinceGpsEpoch"] = acce_df["utcTimeMillis"] - 315964800000
    acce_df["millisSinceGpsEpoch"] = acce_df["millisSinceGpsEpoch"]//1000 +18
    magn_df["millisSinceGpsEpoch"] = magn_df["utcTimeMillis"] - 315964800000
    magn_df["millisSinceGpsEpoch"] = magn_df["millisSinceGpsEpoch"]//1000 +18
    gyro_df["millisSinceGpsEpoch"] = gyro_df["utcTimeMillis"] - 315964800000
    gyro_df["millisSinceGpsEpoch"] = gyro_df["millisSinceGpsEpoch"]//1000 +18
    
    acce_df["x_f_acce"] = acce_df["UncalAccelZMps2"]
    acce_df["y_f_acce"] = acce_df["UncalAccelXMps2"]
    acce_df["z_f_acce"] = acce_df["UncalAccelYMps2"]
    # magn 
    magn_df["x_f_magn"] = magn_df["UncalMagZMicroT"]
    magn_df["y_f_magn"] = magn_df["UncalMagYMicroT"]
    magn_df["z_f_magn"] = magn_df["UncalMagXMicroT"]
    # gyro
    gyro_df["x_f_gyro"] = gyro_df["UncalGyroXRadPerSec"]
    gyro_df["y_f_gyro"] = gyro_df["UncalGyroYRadPerSec"]
    gyro_df["z_f_gyro"] = gyro_df["UncalGyroZRadPerSec"]    

    df = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_bs", "lngDeg_bs", ]].sort_values('millisSinceGpsEpoch'), acce_df[["millisSinceGpsEpoch", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    df = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_bs", "lngDeg_bs", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), magn_df[["millisSinceGpsEpoch", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    df = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_bs", "lngDeg_bs", "x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), gyro_df[["millisSinceGpsEpoch", "x_f_gyro", "y_f_gyro", "z_f_gyro"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    return df



def make_train(INPUT, train_cname, tag_v):
    '''Making the train dataset for modeling. 正式为模型构建训练集。'''
    # make ground_truth file
    gt = make_gt(INPUT, None, None)
    train_df = pd.DataFrame()
    for cname in train_cname:
        phone_list = gt[gt['collectionName'] == cname]['phoneName'].drop_duplicates()
        for pname in phone_list:
            df = gt[(gt['collectionName'] == cname) & (gt['phoneName'] == pname)]
            df = add_IMU(df, INPUT, cname, pname)
            train_df = pd.concat([train_df, df])
    # make tag
    train_df = make_tag(train_df, tag_v)
    return train_df



def make_test(INPUT, test_cname, tag_v):
    '''Making the test dataset for modeling. 正式为模型构建训练集。'''
    # make ground_truth file
    gt = make_gt_tst(INPUT, None, None)
    test_df = pd.DataFrame()
    for cname in test_cname:
        phone_list = gt[gt['collectionName'] == cname]['phoneName'].drop_duplicates()
        for pname in phone_list:
            df = gt[(gt['collectionName'] == cname) & (gt['phoneName'] == pname)]
            df = add_IMU_tst(df, INPUT, cname, pname)
            test_df = pd.concat([test_df, df])
    return test_df



def get_train_score(df):
    '''Calculating the score of the train dataset. 获取训练集的分数。'''
    # calc_distance_error
    df['err'] =  calc_haversine(df.latDeg_bs, df.lngDeg_bs, 
    df.latDeg_gt, df.lngDeg_gt)
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2 
    score = res['p50_p90_mean'].mean()
    return score



def percentile50(x):
    return np.percentile(x, 50)
def percentile95(x):
    return np.percentile(x, 95)
def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    计算地球上两点之间的距离。
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(a**0.5)
    dist = 6_367_000 * c
    return dist



# Define the road type (i.e., highway/street/downtown) for different collections.
# 为训练集和测试集的采集路线定义 路线类型。 
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
'2021-04-22-US-SJC-1':'downtown', 
'2021-04-26-US-SVL-1':'street',
'2021-04-28-US-MTV-1':'street', 
'2021-04-28-US-SJC-1':'downtown',
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



def lgbm(train, test, col, lgb_params):
    '''Build LGBM model. 构建LGBM模型。'''
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(train[col], train['tag'])
    trn_preds = model.predict(train[col])
    tst_preds = model.predict(test[col])
    print('Train:')
    print('confusion matrix :  \n', confusion_matrix(trn_preds, train['tag']))
    print('accuracy score : ', accuracy_score(trn_preds, train['tag']))
    return trn_preds, tst_preds



#  Define Parameter. 参数定义。
INPUT = '../data'
tag_v = 0.5 # Treat the point with Speed < 0.5 as the stopping point. 将速度小于0.5的点当做停车点。
col = ["x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn", "x_f_gyro", "y_f_gyro", "z_f_gyro"]



for road_type in ['downtown', 'street']:
    # Gain the collections under the given road type. 获取各类型路段下的collection。
    train_cname = []
    for col_name, col_type in trn_col_cat_dict.items():
        if col_type == road_type:
            train_cname.append(col_name)
    test_cname = []
    for col_name, col_type in tst_col_cat_dict.items():
        if col_type == road_type:
            test_cname.append(col_name)
    print('train_cname:', train_cname)
    print('test_cname:', test_cname)

    # make train & test. 准备训练集和测试集。
    train_df = make_train(INPUT, train_cname, tag_v)
    test_df = make_test(INPUT, test_cname, tag_v)
    train_df['phone'] = train_df['collectionName'] + '_' + train_df['phoneName']
    test_df['phone'] = test_df['collectionName'] + '_' + test_df['phoneName']

    # different road type, different model params.
    if road_type == 'highway':
        lgb_params = {
                    'num_leaves':22,
                    'n_estimators':95,
                    'random_state':2021,
                    'metric':'accuracy'
                    }
        # For a fraction of stop trajectory, how many points could be detected mistakenly as non-stop points.
        # 容忍停车片段中有多少个点会误检出非停车点。
        unstop_count_limit = 20 
    elif road_type == 'street':
        lgb_params = {
                    'num_leaves':5,
                    'n_estimators':50,
                    'random_state':2021,
                    'metric':'accuracy'
                    }
        unstop_count_limit = 2
    elif road_type == 'downtown':
        lgb_params = {
                    'num_leaves':90,
                    'n_estimators':125,
                    'random_state':2021,
                    'metric':'accuracy'
                    }
        unstop_count_limit = 5
    
    # prediction with lightgbm. LGBM训练及预测。
    train_df['preds'], test_df['preds'] = lgbm(train_df, test_df, col, lgb_params)

    # For the train dataset, gain the car parks; indexes. 针对训练集获取carpark索引。
    for phone in train_df.phone.unique():
        start_point_max_idx = 0
        end_point_min_idx = 0
        tgt_df = train_df[train_df.phone == phone]
        tgt_df.reset_index(drop = True)
        unstop_count = 0
        # we believe the car only stop in the car park area with less than 200s duration.
        # 我们认为车只在停车场停留200s以内时长。
        for i in range(len(tgt_df)):
            if (tgt_df['preds'][i] == 1) and (i < 200) and (unstop_count < unstop_count_limit):
                start_point_max_idx = i + 1
            elif (tgt_df['preds'][i] == 0) and (i < 200) and (unstop_count < unstop_count_limit):
                start_point_max_idx = i + 1
                unstop_count += 1
            else:
                break

        unstop_count = 0
        for i in range(len(tgt_df)):
            if (tgt_df['preds'][len(tgt_df)-i-1] == 1) and (i < 200) and (unstop_count < unstop_count_limit):
                end_point_min_idx = len(tgt_df) - i
            elif (tgt_df['preds'][len(tgt_df)-i-1] == 0) and (i < 200) and (unstop_count < unstop_count_limit):
                end_point_min_idx = len(tgt_df) - i
                unstop_count += 1
            else:
                break
        print("'{}':[{},{}],".format(phone, start_point_max_idx, end_point_min_idx)) 

    # For the test dataset, gain the car parks; indexes. 针对测试集获取carpark索引。
    for phone in test_df.phone.unique():
        start_point_max_idx = 0
        end_point_min_idx = 0

        tgt_df = test_df[test_df.phone == phone]
        tgt_df.reset_index(drop = True)
        unstop_count = 0
        for i in range(len(tgt_df)):
            if (tgt_df['preds'][i] == 1) and (i < 200) and (unstop_count < unstop_count_limit):
                start_point_max_idx = i + 1
            elif (tgt_df['preds'][i] == 0) and (i < 200) and (unstop_count < unstop_count_limit):
                start_point_max_idx = i + 1
                unstop_count += 1
            else:
                break
                
        unstop_count = 0
        for i in range(len(tgt_df)):
            if (tgt_df['preds'][len(tgt_df)-i-1] == 1) and (i < 200) and (unstop_count < unstop_count_limit):
                end_point_min_idx = len(tgt_df) - i
            elif (tgt_df['preds'][len(tgt_df)-i-1] == 0) and (i < 200) and (unstop_count < unstop_count_limit):
                end_point_min_idx = len(tgt_df) - i
                unstop_count += 1
            else:
                break
        print("'{}':[{},{}],".format(phone, start_point_max_idx, end_point_min_idx))  




# Result about the street and downtown collections: 关于street和downtown的预测结果
'''
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
'''
                    