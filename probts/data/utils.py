# ---------------------------------------------------------------------------------
# Portions of this file are derived from Autoformer
# - Source: https://github.com/thuml/Autoformer/tree/main
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import os
import warnings

import numpy as np
import pandas as pd

from .time_features import time_features

warnings.filterwarnings("ignore")


def get_LTSF_info(dataset):
    if dataset == "etth1" or dataset == "etth2":
        if dataset == "etth1":
            data_path = "ETT-small/ETTh1.csv"
        else:
            data_path = "ETT-small/ETTh2.csv"
        freq = "H"
    elif dataset == "ettm1" or dataset == "ettm2":
        if dataset == "ettm1":
            data_path = "ETT-small/ETTm1.csv"
        else:
            data_path = "ETT-small/ETTm2.csv"
        freq = "min"
    elif dataset in [
        "traffic_ltsf",
        "electricity_ltsf",
        "exchange_ltsf",
        "illness_ltsf",
        "weather_ltsf",
    ]:
        if dataset == "traffic_ltsf":
            data_path = "traffic/traffic.csv"
            freq = "H"
        elif dataset == "electricity_ltsf":
            data_path = "electricity/electricity.csv"
            freq = "H"
        elif dataset == "exchange_ltsf":
            data_path = "exchange_rate/exchange_rate.csv"
            freq = "B"
        elif dataset == "illness_ltsf":
            data_path = "illness/national_illness.csv"
            freq = "W"
        elif dataset == "weather_ltsf":
            data_path = "weather/weather.csv"
            freq = "min"
    elif dataset == "caiso":
        data_path = "caiso/caiso_20130101_20210630.csv"
        freq = "H"
    elif dataset == "nordpool":
        data_path = "nordpool/production.csv"
        freq = "H"
    else:
        raise ValueError(f"Invalid dataset name: {dataset}!")
    return data_path, freq


def get_LTSF_borders(dataset, data_size):
    if dataset == "etth1" or dataset == "etth2":
        border_begin = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border_end = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
    elif dataset == "ettm1" or dataset == "ettm2":
        border_begin = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        border_end = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
    else:
        num_train = int(data_size * 0.7)
        num_test = int(data_size * 0.2)
        num_vali = data_size - num_train - num_test
        border_begin = [0, num_train, data_size - num_test]
        border_end = [num_train, num_train + num_vali, data_size]
    return border_begin, border_end


def get_LTSF_Dataset(root_path, data_path, freq="h", timeenc=1):
    if "caiso" in data_path:
        data = pd.read_csv(root_path + data_path)
        data["Date"] = data["Date"].astype("datetime64[ns]")
        names = [
            "PGE",
            "SCE",
            "SDGE",
            "VEA",
            "CA ISO",
            "PACE",
            "PACW",
            "NEVP",
            "AZPS",
            "PSEI",
        ]
        df_raw = pd.DataFrame(
            pd.date_range("20130101", "20210630", freq="H")[:-1], columns=["Date"]
        )
        for name in names:
            current_df = (
                data[data["zone"] == name]
                .drop_duplicates(subset="Date", keep="last")
                .rename(columns={"load": name})
                .drop(columns=["zone"])
            )
            df_raw = df_raw.merge(current_df, on="Date", how="outer")
        df_raw = df_raw.rename(columns={"Date": "date"})
    elif "nordpool" in data_path:
        df_raw = pd.read_csv(root_path + data_path, parse_dates=["Time"])
        df_raw = df_raw.rename(columns={"Time": "date"})
    else:
        df_raw = pd.read_csv(os.path.join(root_path, data_path))

    df_stamp = df_raw[["date"]]
    df_stamp["date"] = pd.to_datetime(df_stamp.date)

    if timeenc == 0:
        df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
        data_stamp = df_stamp.drop(labels="date", axis=1).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    elif timeenc == 2:
        data_stamp = pd.to_datetime(df_stamp["date"].values)
        data_stamp = np.array(data_stamp, dtype="datetime64[s]")

    df_raw = df_raw.set_index(keys="date")
    df_raw = df_raw.fillna(0)
    target_dim = len(df_raw.columns)
    data_size = len(df_raw)
    return df_raw, data_stamp, target_dim, data_size


# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#         border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], axis=1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_ETT_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTm1.csv',
#                  target='OT', scale=True, timeenc=0, freq='t'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
#         border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], axis=1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         # print(cols)
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             # print(self.scaler.mean_)
#             # exit()
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], axis=1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_Pred(Dataset):
#     def __init__(self, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['pred']

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols = cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         if self.cols:
#             cols = self.cols.copy()
#             cols.remove(self.target)
#         else:
#             cols = list(df_raw.columns)
#             cols.remove(self.target)
#             cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         border1 = len(df_raw) - self.seq_len
#         border2 = len(df_raw)

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             self.scaler.fit(df_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         tmp_stamp = df_raw[['date']][border1:border2]
#         tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
#         pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

#         df_stamp = pd.DataFrame(columns=['date'])
#         df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], axis=1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         if self.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         if self.inverse:
#             seq_y = self.data_x[r_begin:r_begin + self.label_len]
#         else:
#             seq_y = self.data_y[r_begin:r_begin + self.label_len]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
