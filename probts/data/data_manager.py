import torch
from copy import deepcopy
from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository import dataset_names, datasets
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from .ltsf_datasets import get_LTSF_info, get_LTSF_borders, get_LTSF_Dataset
from .stsf_datasets import GluonTSDataset
from .time_features import get_lags

from probts.utils import StandardScaler, TemporalScaler, IdentityScaler


MULTI_VARIATE_DATASETS = [
    'exchange_rate_nips',
    'solar_nips',
    'electricity_nips',
    'traffic_nips',
    'taxi_30min',
    'wiki-rolling_nips',
    'wiki2000_nips'
]


class ProbTSBatchData:
    input_names_ = [
        'target_dimension_indicator',
        'past_time_feat',
        'past_target_cdf',
        'past_observed_values',
        'past_is_pad',
        'future_time_feat',
        'future_target_cdf',
        'future_observed_values',
    ]
    
    def __init__(self, data_dict, device):
        self.__dict__.update(data_dict)
        if len(self.__dict__['past_target_cdf'].shape) == 2:
            self.expand_dim()
        self.set_device(device)
        self.fill_inputs()
        self.process_pad()

    def fill_inputs(self):
        for input in self.input_names_:
            if input not in self.__dict__:
                self.__dict__[input] = None

    def set_device(self, device):
        for k, v in self.__dict__.items():
            if v is not None:
                v.to(device)
        self.device = device

    def expand_dim(self):
        self.__dict__["target_dimension_indicator"] = self.__dict__["target_dimension_indicator"][:, :1]
        for input in ['past_target_cdf','past_observed_values','future_target_cdf','future_observed_values']:
            self.__dict__[input] = self.__dict__[input].unsqueeze(-1)

    def process_pad(self):
        if self.__dict__['past_is_pad'] is not None:
            self.__dict__["past_observed_values"] = torch.min(
                self.__dict__["past_observed_values"],
                1 - self.__dict__["past_is_pad"].unsqueeze(-1)
            )


class DataManager:

    def __init__(
        self,
        dataset,
        path: str = './datasets',
        history_length: int = None,
        context_length: int = None,
        prediction_length: int = None,
        test_rolling_length: int = 96,
        split_val: bool = True,
        scaler: str = 'none',
        test_sampling: str = 'ctx', # ['arrange', 'ctx', 'pred']
        context_length_factor: int = 1,
        timeenc: int = 1,
        var_specific_norm: bool = True,
    ):
        self.dataset = dataset
        self.test_rolling_length = test_rolling_length
        self.global_mean = None
        self.multivariate = True 
        self.split_val = split_val
        self.test_sampling = test_sampling
        self.timeenc = timeenc
        self.var_specific_norm = var_specific_norm

        if scaler == 'standard':
            self.scaler = StandardScaler(var_specific=self.var_specific_norm)
        elif scaler == 'temporal':
            self.scaler = TemporalScaler()
        else:
            self.scaler = IdentityScaler()
    
        if dataset in dataset_names:
            """
            Use gluonts to load short-term datasets.
            """
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

            data_path, self.freq = get_LTSF_info(dataset)
            self.dataset_raw, self.data_stamp, self.target_dim, data_size = get_LTSF_Dataset(path, data_path,freq=self.freq,timeenc=self.timeenc)
            self.border_begin, self.border_end = get_LTSF_borders(dataset, data_size)

            assert data_size >= self.border_end[2], print("\n The end index larger then data size!")

            # Meta parameters
            self.lags_list = get_lags(self.freq)
            self.prediction_length = prediction_length
            self.context_length = context_length
            self.history_length = (self.context_length + max(self.lags_list)) \
                if history_length is None else history_length

            self.prepare_LTSF_dataset()

        print(f"context_length: {self.context_length}, prediction_length: {self.prediction_length}")
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
        
        group_val_set = self.get_rolling_test(group_val_set, self.border_begin[1], self.border_end[1], rolling_length=self.test_rolling_length)
        group_test_set = self.get_rolling_test(group_test_set, self.border_begin[2], self.border_end[2], rolling_length=self.test_rolling_length)
        
        stsf_dataset_loader = GluonTSDataset(
            ProbTSBatchData.input_names_, 
            self.history_length,
            self.prediction_length,
            self.freq,
            self.multivariate
        )
        
        self.train_iter_dataset = stsf_dataset_loader.get_iter_dataset(group_train_set, mode='train', data_stamp=self.data_stamp[: self.border_end[0]])
        self.val_iter_dataset = stsf_dataset_loader.get_iter_dataset(group_val_set, mode='val', data_stamp=self.data_stamp[: self.border_end[1]])
        self.test_iter_dataset = stsf_dataset_loader.get_iter_dataset(group_test_set, mode='test', data_stamp=self.data_stamp[: self.border_end[2]])
        self.time_feat_dim = stsf_dataset_loader.time_feat_dim
        self.global_mean = torch.mean(torch.tensor(group_train_set[0]['target']), dim=-1)

    def df_to_mvds(self, df, freq='H'):
        datasets = []
        for variable in df.keys():
            ds = {"item_id" : variable, "target" : df[variable], "start": str(df.index[0])}
            datasets.append(ds)
        dataset = ListDataset(datasets,freq=freq)
        return dataset

    def get_rolling_test(self, test_set, border_begin_idx, border_end_idx, rolling_length):
        num_test_dates = int(((border_end_idx - border_begin_idx - self.prediction_length) / rolling_length))
        print("num_test_dates: ", num_test_dates)

        test_set = next(iter(test_set))
        rolling_test_seq_list = list()
        for i in range(num_test_dates):
            rolling_test_seq = deepcopy(test_set)
            rolling_end = border_begin_idx + self.prediction_length + i * rolling_length
            rolling_test_seq[FieldName.TARGET] = rolling_test_seq[FieldName.TARGET][:, :rolling_end]
            rolling_test_seq_list.append(rolling_test_seq)

        rolling_test_set = ListDataset(
            rolling_test_seq_list, freq=self.freq, one_dim_target=False
        )
        return rolling_test_set

    def split_train_val(self, train_set):
        trunc_train_list = []
        val_set_list = []
        univariate = False

        for train_seq in iter(train_set):
            # truncate train set
            offset = self.num_test_dates * self.prediction_length
            trunc_train_seq = deepcopy(train_seq)

            if len(train_seq[FieldName.TARGET].shape) == 1:
                trunc_train_len = train_seq[FieldName.TARGET].shape[0] - offset
                trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][:trunc_train_len]
                univariate = True
            elif len(train_seq[FieldName.TARGET].shape) == 2:
                trunc_train_len = train_seq[FieldName.TARGET].shape[1] - offset
                trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][:, :trunc_train_len]
            else:
                raise ValueError(f"Invalid Data Shape: {str(len(train_seq[FieldName.TARGET].shape))}")

            trunc_train_list.append(trunc_train_seq)

            # construct val set by rolling
            for i in range(self.num_test_dates):
                val_seq = deepcopy(train_seq)
                rolling_len = trunc_train_len + self.prediction_length * (i+1)
                if univariate:
                    val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][trunc_train_len + self.prediction_length * (i-1) - self.context_length : rolling_len]
                else:
                    val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][:, :rolling_len]
                
                val_set_list.append(val_seq)

        trunc_train_set = ListDataset(
            trunc_train_list, freq=self.freq, one_dim_target=univariate
        )

        val_set = ListDataset(
            val_set_list, freq=self.freq, one_dim_target=univariate
        )
        
        return trunc_train_set, val_set

    def truncate_test(self, test_set):
        trunc_test_list = []
        for test_seq in iter(test_set):
            # truncate train set
            trunc_test_seq = deepcopy(test_seq)

            trunc_test_seq[FieldName.TARGET] = trunc_test_seq[FieldName.TARGET][- ( self.prediction_length * 2 + self.context_length):]

            trunc_test_list.append(trunc_test_seq)

        trunc_test_set = ListDataset(
            trunc_test_list, freq=self.freq, one_dim_target=True
        )

        return trunc_test_set
    
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
            test_set = self.truncate_test(test_set)

        if self.split_val:
            train_set, val_set = self.split_train_val(train_set)
        else:
            val_set = test_set

        stsf_dataset_loader = GluonTSDataset(
            ProbTSBatchData.input_names_, 
            self.history_length,
            self.prediction_length,
            self.freq,
            self.multivariate
        )

        self.train_iter_dataset = stsf_dataset_loader.get_iter_dataset(train_set, mode='train')
        self.val_iter_dataset = stsf_dataset_loader.get_iter_dataset(val_set, mode='val')
        self.test_iter_dataset = stsf_dataset_loader.get_iter_dataset(test_set, mode='test')
        self.time_feat_dim = stsf_dataset_loader.time_feat_dim
