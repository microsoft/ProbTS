import torch
from copy import deepcopy
from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from .ltsf_datasets import get_LTSF_info, get_LTSF_borders, get_LTSF_Dataset
from .time_features import get_lags
from .data_wrapper import ProbTSBatchData
from .multi_horizon_dataset import MultiHorizonDataset
from probts.utils import StandardScaler, TemporalScaler, IdentityScaler, convert_to_list
from typing import Union
import math

MULTI_VARIATE_DATASETS = [
    'exchange_rate_nips',
    'solar_nips',
    'electricity_nips',
    'traffic_nips',
    'taxi_30min',
    'wiki-rolling_nips',
    'wiki2000_nips'
]

class DataManager:

    def __init__(
        self,
        dataset,
        path: str = './datasets',
        history_length: int = None,
        context_length: Union[list,int,str] = None,
        prediction_length: Union[list,int,str] = None,
        train_ctx_len_list: Union[list,int,str] = None,
        train_pred_len_list: Union[list,int,str] = None,
        val_pred_len_list: Union[list,int,str] = None,
        val_ctx_len_list: Union[list,int,str] = None,
        test_rolling_length: int = None,
        split_val: bool = True,
        scaler: str = 'none',
        test_sampling: str = 'ctx', # ['arrange', 'ctx', 'pred']
        timeenc: int = 1,
        var_specific_norm: bool = True,
        continous_sample: bool = False,
    ):
        self.dataset = dataset
        self.multivariate = True 
        self.split_val = split_val
        self.test_sampling = test_sampling
        self.timeenc = timeenc
        self.var_specific_norm = var_specific_norm
        self.continous_sample = continous_sample
        self.test_rolling_length = test_rolling_length
        
        self.test_rolling_dict = {'h': 24, 'd': 7, 'b':5, 'w':4, 'min': 60}

        if scaler == 'standard':
            self.scaler = StandardScaler(var_specific=self.var_specific_norm)
        elif scaler == 'temporal':
            self.scaler = TemporalScaler()
        else:
            self.scaler = IdentityScaler()

        # multi-horizon training
        self.pred_len_list = convert_to_list(train_pred_len_list)
        if self.pred_len_list is None:
            self.pred_len_list = convert_to_list(prediction_length)
        
        self.ctx_len_list = convert_to_list(train_ctx_len_list)
        if self.ctx_len_list is None:
            self.ctx_len_list = convert_to_list(context_length)
            
            
        self.val_pred_len_list = convert_to_list(val_pred_len_list)
        if self.val_pred_len_list is None:
            self.val_pred_len_list = self.pred_len_list
        
        self.val_ctx_len_list = convert_to_list(val_ctx_len_list)
        if self.val_ctx_len_list is None:
            self.val_ctx_len_list = self.ctx_len_list
    

        # Load datasets.
        print("Loading Long-term Datasets: {dataset}".format(dataset=dataset))

        if context_length is None or prediction_length is None:
            raise ValueError("The context_length or prediction_length is not assigned.")

        data_path, self.freq = get_LTSF_info(dataset)
        self.dataset_raw, self.data_stamp, self.target_dim, data_size = get_LTSF_Dataset(path, data_path,freq=self.freq,timeenc=self.timeenc)
        self.border_begin, self.border_end = get_LTSF_borders(dataset, data_size)

        assert data_size >= self.border_end[2], print("\n The end index larger then data size!")

        # Meta parameters
        self.lags_list = get_lags(self.freq)
        self.prediction_length = convert_to_list(prediction_length)
        self.context_length = convert_to_list(context_length)
        self.history_length = (max(self.context_length) + max(self.lags_list)) \
            if history_length is None else history_length
            
        # define the test_rolling_length
        if self.test_rolling_length is None:
            if self.freq.lower() in self.test_rolling_dict:
                self.test_rolling_length = self.test_rolling_dict[self.freq.lower()]
            else:
                self.test_rolling_length = 24

        self.prepare_LTSF_dataset()

        print(f"test context_length: {self.context_length}, test prediction_length: {self.prediction_length}")
        print(f"training context lengths: {self.ctx_len_list}, training prediction lengths: {self.pred_len_list}")
        print(f"Sampling T from [{min(self.pred_len_list)},  {max(self.pred_len_list)}], Continuous: {self.continous_sample}")
        print(f"test_rolling_length: {self.test_rolling_length}")

        if scaler == 'standard':
            print(f"variate-specific normalization: {self.var_specific_norm}")

    def prepare_LTSF_dataset(self):
        train_data = self.dataset_raw[: self.border_end[0]]
        val_data = self.dataset_raw[: self.border_end[1]]
        test_data = self.dataset_raw[: self.border_end[2]]
        
        self.scaler.fit(torch.tensor(train_data.values))
        
        train_set = self.df_to_mvds(train_data, freq=self.freq)
        val_set = self.df_to_mvds(val_data,freq=self.freq)
        test_set = self.df_to_mvds(test_data,freq=self.freq)
        
        train_grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        test_grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        
        group_train_set = train_grouper(train_set)
        group_val_set = test_grouper(val_set)
        group_test_set = test_grouper(test_set)
        
        dataset_loader = MultiHorizonDataset(ProbTSBatchData.input_names_, 
            self.context_length,
            self.prediction_length,
            self.freq,
            self.ctx_len_list,
            self.pred_len_list,
            self.val_ctx_len_list,
            self.val_pred_len_list,
            self.multivariate,
            self.continous_sample)
        
        self.train_iter_dataset = dataset_loader.get_iter_dataset(group_train_set, mode='train', data_stamp=self.data_stamp[: self.border_end[0]])
        
        # for multi-horizon evaluation
        self.val_iter_dataset = {}
        self.test_iter_dataset = {}

        # validation set
        for pred_len in self.val_pred_len_list:
            local_group_val_set = self.get_rolling_test('val', group_val_set, self.border_begin[1], self.border_end[1], rolling_length=self.test_rolling_length, pred_len=pred_len)
            self.val_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(local_group_val_set, mode='val', data_stamp=self.data_stamp[: self.border_end[1]], pred_len=[pred_len])
            
        # testing set
        for pred_len in self.prediction_length:
            local_group_test_set = self.get_rolling_test('test', group_test_set, self.border_begin[2], self.border_end[2], rolling_length=self.test_rolling_length, pred_len=pred_len)
            self.test_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(local_group_test_set, mode='test', data_stamp=self.data_stamp[: self.border_end[2]], pred_len=[pred_len])
        
        self.time_feat_dim = dataset_loader.time_feat_dim

    def df_to_mvds(self, df, freq='H'):
        datasets = []
        for variable in df.keys():
            ds = {"item_id" : variable, "target" : df[variable], "start": str(df.index[0])}
            datasets.append(ds)
        dataset = ListDataset(datasets,freq=freq)
        return dataset

    def get_rolling_test(self, stage, test_set, border_begin_idx, border_end_idx, rolling_length, pred_len):
        num_test_dates = math.ceil(((border_end_idx - border_begin_idx - pred_len) / rolling_length))
        print(f"{stage}  pred_len: {pred_len} : num_test_dates: {num_test_dates}")

        test_set = next(iter(test_set))
        rolling_test_seq_list = list()
        for i in range(num_test_dates):
            rolling_test_seq = deepcopy(test_set)
            rolling_end = border_begin_idx + pred_len + i * rolling_length
            rolling_test_seq[FieldName.TARGET] = rolling_test_seq[FieldName.TARGET][:, :rolling_end]
            rolling_test_seq_list.append(rolling_test_seq)

        rolling_test_set = ListDataset(
            rolling_test_seq_list, freq=self.freq, one_dim_target=False
        )
        return rolling_test_set


