# ---------------------------------------------------------------------------------
# Portions of this file are derived from Autoformer
# - Source: https://github.com/thuml/Autoformer/tree/main
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import os
import pandas as pd
from .time_features import time_features
from .monash_datasets import convert_monash_data_to_dataframe, monash_format_convert
import numpy as np

def get_LTSF_info(dataset):
    if dataset == 'etth1' or dataset == 'etth2':
        if dataset == 'etth1':
            data_path = 'ETT-small/ETTh1.csv'
        else:
            data_path = 'ETT-small/ETTh2.csv'
        freq = 'h'
    elif dataset == 'ettm1' or dataset == 'ettm2':
        if dataset == 'ettm1':
            data_path = 'ETT-small/ETTm1.csv'
        else:
            data_path = 'ETT-small/ETTm2.csv'
        freq = 'min'
    elif dataset in ['traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf']:
        if dataset == 'traffic_ltsf':
            data_path = 'traffic/traffic.csv'
            freq = 'h'
        elif dataset == 'electricity_ltsf':
            data_path = 'electricity/electricity.csv'
            freq = 'h'
        elif dataset == 'exchange_ltsf':
            data_path = 'exchange_rate/exchange_rate.csv'
            freq = 'B'
        elif dataset == 'illness_ltsf':
            data_path = 'illness/national_illness.csv'
            freq = 'W'
        elif dataset == 'weather_ltsf':
            data_path = 'weather/weather.csv'
            freq = 'min'
    elif dataset == 'caiso':
        data_path = 'caiso/caiso_20130101_20210630.csv'
        freq = 'h'
    elif dataset == 'nordpool':
        data_path = 'nordpool/production.csv'
        freq = 'h'
    elif dataset == 'solar_hour':
        data_path = 'monash/solar_10_minutes_dataset.tsf'
        freq = 'h'
    elif dataset == 'solar_10min':
        data_path = 'monash/solar_10_minutes_dataset.tsf'
        freq = 'min'
    elif dataset == 'sunspot':
        data_path = 'monash/sunspot_dataset_without_missing_values.tsf'
        freq = 'D'
    elif dataset == 'river_flow':
        data_path = 'monash/saugeenday_dataset.tsf'
        freq = 'D'
    else:
        raise ValueError(f"Invalid dataset name: {dataset}!")
    return data_path, freq

def get_LTSF_borders(dataset, data_size):
    if dataset == 'etth1' or dataset == 'etth2':
        border_begin = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border_end = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset == 'ettm1' or dataset == 'ettm2':
        border_begin = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        border_end = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(data_size * 0.7)
        num_test = int(data_size * 0.2)
        num_vali = data_size - num_train - num_test
        border_begin = [0, num_train, data_size - num_test]
        border_end = [num_train, num_train + num_vali, data_size]
    return border_begin, border_end

def get_LTSF_Dataset(root_path, data_path, freq='h', timeenc=1, multivariate=True):
    if 'caiso' in data_path:
        data = pd.read_csv(root_path + data_path)
        data['Date'] = data['Date'].astype('datetime64[ns]')
        names = ['PGE','SCE','SDGE','VEA','CA ISO','PACE','PACW','NEVP','AZPS','PSEI']
        df_raw = pd.DataFrame(pd.date_range('20130101','20210630',freq='H')[:-1], columns=['Date'])
        for name in names:
            current_df = data[data['zone'] == name].drop_duplicates(subset='Date', keep='last').rename(columns={'load':name}).drop(columns=['zone'])
            df_raw = df_raw.merge(current_df, on='Date', how='outer')
        df_raw = df_raw.rename(columns={'Date': 'date'})
    elif 'nordpool' in data_path:
        df_raw = pd.read_csv(root_path + data_path, parse_dates=['Time'])
        df_raw = df_raw.rename(columns={'Time': 'date'})
    elif '.tsf' in data_path:
        df_raw, _, _, _, _ = convert_monash_data_to_dataframe(data_path)
        df_raw = monash_format_convert(df_raw, freq, multivariate)
        
        if multivariate:
            if freq.lower() == 'h':
                df_raw.set_index('date', inplace=True)
                df_raw = df_raw.resample(freq).mean().reset_index()
    else:
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
    
    if multivariate:
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(labels='date', axis=1).values
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
            data_stamp = data_stamp.transpose(1, 0)
        elif timeenc == 2:
            data_stamp = pd.to_datetime(df_stamp['date'].values)
            data_stamp = np.array(data_stamp, dtype='datetime64[s]')

        df_raw = df_raw.set_index(keys='date')
        
    else:
        data_stamp = None
        
    df_raw = df_raw.fillna(0)
    target_dim = len(df_raw.columns) if multivariate else 1
    data_size = len(df_raw)
    return df_raw, data_stamp, target_dim, data_size