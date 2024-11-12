import torch
from pathlib import Path

from gluonts.dataset.repository import dataset_names, datasets
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from probts.data.datasets.ltsf_datasets import get_LTSF_info, get_LTSF_borders, get_LTSF_Dataset
from probts.data.datasets.stsf_datasets import GluonTSDataset
from probts.data.datasets.multi_horizon_datasets import MultiHorizonDataset

from probts.data.data_utils.time_features import get_lags
from probts.data.data_utils.data_utils import split_train_val, truncate_test, get_rolling_test, df_to_mvds
from probts.data.data_wrapper import ProbTSBatchData
from probts.utils.utils import ensure_list
from probts.data.data_utils.data_scaler import StandardScaler, TemporalScaler, IdentityScaler
from typing import Union
import sys

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
        train_ctx_len: Union[list,int,str] = None,
        train_pred_len_list: Union[list,int,str] = None,
        val_ctx_len: Union[list,int,str] = None,
        val_pred_len_list: Union[list,int,str] = None,
        test_rolling_length: int = 96,
        split_val: bool = True,
        scaler: str = 'none',
        test_sampling: str = 'ctx', # ['arrange', 'ctx', 'pred']
        context_length_factor: int = 1,
        timeenc: int = 1,
        var_specific_norm: bool = True,
        data_path: str = None,
        freq: str = None,
        multivariate: bool = True,
        continuous_sample: bool = False,
    ):
        self.dataset = dataset
        self.global_mean = None
        self.multivariate = multivariate 
        self.split_val = split_val
        self.test_sampling = test_sampling
        self.timeenc = timeenc
        self.var_specific_norm = var_specific_norm
        self.continuous_sample = continuous_sample
        self.test_rolling_length = test_rolling_length
        self.test_rolling_dict = {'h': 24, 'd': 7, 'b':5, 'w':4, 'min': 60}

        # Global Normalization
        if scaler == 'standard':
            self.scaler = StandardScaler(var_specific=self.var_specific_norm)
        elif scaler == 'temporal':
            self.scaler = TemporalScaler()
        else:
            self.scaler = IdentityScaler()

        # covert str to list to support multi-horizon training & inference
        self.train_pred_len_list = ensure_list(train_pred_len_list, default_value=prediction_length)
        self.val_pred_len_list = ensure_list(val_pred_len_list, default_value=prediction_length)
        self.test_pred_len_list = ensure_list(prediction_length)
        
        self.train_ctx_len_list = ensure_list(train_ctx_len, default_value=context_length)
        self.val_ctx_len_list = ensure_list(val_ctx_len, default_value=context_length)
        self.test_ctx_len_list = ensure_list(context_length)
        
        # We do not support multi-context so far, but context can be varied across train/validation/testing splits
        assert (len(self.train_ctx_len_list) == 1) and (len(self.val_ctx_len_list) == 1) and (len(self.test_ctx_len_list) == 1), \
            "We do not support multi-context so far, but context can be varied across train/validation/testing splits. Assign a single context length for each split."
        
        self.multi_hor = True
        
        # if multiple horizons assigned, using MultiHorizonDataset
        if prediction_length is None:
            self.multi_hor = False
        else:
            if len(self.train_pred_len_list) == 1 and len(self.val_pred_len_list) == 1 and len(self.test_pred_len_list) == 1:
                if self.val_pred_len_list[0] == self.test_pred_len_list[0] and self.train_pred_len_list[0] == self.test_pred_len_list[0]:
                    self.multi_hor = False
            
            
        if dataset in dataset_names:
            """
            Use gluonts to load short-term datasets.
            """
            if self.multi_hor:
                raise ValueError("Short-term forecasting do not support multi-horizon training or inference. ]")

            print("Loading Short-term Datasets: {dataset}".format(dataset=dataset))
            self.dataset_raw = datasets.get_dataset(dataset, path=Path(path), regenerate=False)

            # Meta parameters
            self.target_dim = int(self.dataset_raw.metadata.feat_static_cat[0].cardinality)
            self.freq = self.dataset_raw.metadata.freq.upper()
            self.lags_list = get_lags(self.freq)
            self.prediction_length = self.dataset_raw.metadata.prediction_length \
                if prediction_length is None else prediction_length
            self.context_length = self.dataset_raw.metadata.prediction_length * context_length_factor \
                if context_length is None else context_length
            self.history_length = (self.context_length + max(self.lags_list)) \
                if history_length is None else history_length

            self.prepare_STSF_dataset(dataset)
        else:
            """
            Load long-term datasets.
            """
            print("Loading Long-term Datasets: {dataset}".format(dataset=dataset))

            if context_length is None or prediction_length is None:
                raise ValueError("The context_length or prediction_length is not assigned.")

            data_path, self.freq = get_LTSF_info(dataset, data_path=data_path, freq=freq)
            self.dataset_raw, self.data_stamp, self.target_dim, data_size = get_LTSF_Dataset(path, data_path,freq=self.freq,timeenc=self.timeenc, multivariate=self.multivariate)
            self.border_begin, self.border_end = get_LTSF_borders(dataset, data_size)
            
            if not self.multivariate:
                self.target_dim = 1
                raise NotImplementedError("Support for customized univariate dataset is still work in progress.")
                
            assert data_size >= self.border_end[2], print("\n The end index larger then data size!")

            # Meta parameters
            self.lags_list = get_lags(self.freq)
            if self.multi_hor:
                self.prediction_length = ensure_list(prediction_length)
                self.context_length = ensure_list(context_length)
                self.history_length = (max(self.context_length) + max(self.lags_list)) \
                    if history_length is None else history_length
            else:
                self.prediction_length = prediction_length
                self.context_length = context_length
                self.history_length = (self.context_length + max(self.lags_list)) \
                    if history_length is None else history_length

            # define the test_rolling_length
            if self.test_rolling_length is None:
                if self.freq.lower() in self.test_rolling_dict:
                    self.test_rolling_length = self.test_rolling_dict[self.freq.lower()]
                else:
                    self.test_rolling_length = 24
            self.prepare_LTSF_dataset()

        print(f"test context length: {self.test_ctx_len_list}, test prediction length: {self.test_pred_len_list}")
        print(f"validation context length: {self.val_ctx_len_list}, validation prediction length: {self.val_pred_len_list}")
        print(f"training context lengths: {self.train_ctx_len_list}, training prediction lengths: {self.train_pred_len_list}")
        print(f"test rolling length: {self.test_rolling_length}")
        if scaler == 'standard':
            print(f"variate-specific normalization: {self.var_specific_norm}")

    def prepare_LTSF_dataset(self):
        train_data = self.dataset_raw[: self.border_end[0]]
        val_data = self.dataset_raw[: self.border_end[1]]
        test_data = self.dataset_raw[: self.border_end[2]]
        
        self.scaler.fit(torch.tensor(train_data.values))
        
        train_set = df_to_mvds(train_data, freq=self.freq)
        val_set = df_to_mvds(val_data,freq=self.freq)
        test_set = df_to_mvds(test_data,freq=self.freq)
        
        train_grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        test_grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        
        group_train_set = train_grouper(train_set)
        group_val_set = test_grouper(val_set)
        group_test_set = test_grouper(test_set)
        
        if self.multi_hor:
            self.val_iter_dataset = {}
            self.test_iter_dataset = {}
            dataset_loader = MultiHorizonDataset(
                input_names = ProbTSBatchData.input_names_,
                freq = self.freq,
                train_ctx_range = self.train_ctx_len_list,
                train_pred_range = self.train_pred_len_list,
                val_ctx_range = self.val_ctx_len_list,
                val_pred_range = self.val_pred_len_list,
                test_ctx_range = self.test_ctx_len_list,
                test_pred_range = self.test_pred_len_list,
                multivariate = self.multivariate,
                continuous_sample = self.continuous_sample
            )
            # validation set
            for pred_len in self.val_pred_len_list:
                local_group_val_set = get_rolling_test('val', group_val_set, self.border_begin[1], self.border_end[1], rolling_length=self.test_rolling_length, pred_len=pred_len, freq=self.freq)
                self.val_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(local_group_val_set, mode='val', data_stamp=self.data_stamp[: self.border_end[1]], pred_len=[pred_len])
                
            # testing set
            for pred_len in self.test_pred_len_list:
                local_group_test_set = get_rolling_test('test', group_test_set, self.border_begin[2], self.border_end[2], rolling_length=self.test_rolling_length, pred_len=pred_len, freq=self.freq)
                self.test_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(local_group_test_set, mode='test', data_stamp=self.data_stamp[: self.border_end[2]], pred_len=[pred_len])
        
        else:
            dataset_loader = GluonTSDataset(
                ProbTSBatchData.input_names_, 
                self.history_length,
                self.prediction_length,
                self.freq,
                self.multivariate
            )
            # validation set
            local_group_val_set = get_rolling_test('val', group_val_set, self.border_begin[1], self.border_end[1], rolling_length=self.test_rolling_length, pred_len=self.val_pred_len_list[0], freq=self.freq)
            self.val_iter_dataset = dataset_loader.get_iter_dataset(local_group_val_set, mode='val', data_stamp=self.data_stamp[: self.border_end[1]])
            # testing set
            local_group_test_set = get_rolling_test('test', group_test_set, self.border_begin[2], self.border_end[2], rolling_length=self.test_rolling_length, pred_len=self.prediction_length, freq=self.freq)
            self.test_iter_dataset = dataset_loader.get_iter_dataset(local_group_test_set, mode='test', data_stamp=self.data_stamp[: self.border_end[2]])
        
        self.train_iter_dataset = dataset_loader.get_iter_dataset(group_train_set, mode='train', data_stamp=self.data_stamp[: self.border_end[0]])
        
        self.time_feat_dim = dataset_loader.time_feat_dim
        self.global_mean = torch.mean(torch.tensor(group_train_set[0]['target']), dim=-1)
    
    def prepare_STSF_dataset(self, dataset: str):
        if dataset in MULTI_VARIATE_DATASETS:
            self.num_test_dates = int(len(self.dataset_raw.test)/len(self.dataset_raw.train))

            train_grouper = MultivariateGrouper(max_target_dim=int(self.target_dim))
            test_grouper = MultivariateGrouper(
                num_test_dates=self.num_test_dates, 
                max_target_dim=int(self.target_dim)
            )
            train_set = train_grouper(self.dataset_raw.train)
            test_set = test_grouper(self.dataset_raw.test)
            self.scaler.fit(torch.tensor(train_set[0]['target'].transpose(1, 0)))
            self.global_mean = torch.mean(torch.tensor(train_set[0]['target']), dim=-1)
        else:
            self.target_dim = 1
            self.multivariate = False
            self.num_test_dates = 1
            train_set = self.dataset_raw.train
            test_set = self.dataset_raw.test
            test_set = truncate_test(test_set)
            
        if self.split_val:
            train_set, val_set = split_train_val(train_set, self.num_test_dates, self.context_length, self.prediction_length, self.freq)
        else:
            val_set = test_set

        dataset_loader = GluonTSDataset(
            ProbTSBatchData.input_names_, 
            self.history_length,
            self.prediction_length,
            self.freq,
            self.multivariate
        )

        self.train_iter_dataset = dataset_loader.get_iter_dataset(train_set, mode='train')
        self.val_iter_dataset = dataset_loader.get_iter_dataset(val_set, mode='val')
        self.test_iter_dataset = dataset_loader.get_iter_dataset(test_set, mode='test')
        self.time_feat_dim = dataset_loader.time_feat_dim
