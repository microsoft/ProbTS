# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import torch
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository import datasets

from .probts_datasets import ProbTSDataset
from .time_features import get_lags

MULTI_VARIATE_DATASETS = [
    "exchange_rate_nips",
    "solar_nips",
    "electricity_nips",
    "traffic_nips",
    "taxi_30min",
    "wiki-rolling_nips",
    "wiki2000_nips",
]


@dataclass
class GluonTSDatasetLoader(ProbTSDataset):
    context_length_factor: int = field(default=1)

    def __post_init__(self):
        super().__post_init__()

        multivariate = True
        split_val = True

        self.__read_data(self.dataset, self.path)
        # self.prepare_STSF_dataset(dataset)

        if self.dataset in MULTI_VARIATE_DATASETS:
            self.num_test_dates = int(
                len(self.dataset_raw.test) / len(self.dataset_raw.train)
            )

            train_grouper = MultivariateGrouper(max_target_dim=int(self.target_dim))
            test_grouper = MultivariateGrouper(
                num_test_dates=self.num_test_dates, max_target_dim=int(self.target_dim)
            )
            self.train_set = train_grouper(self.dataset_raw.train)
            self.test_set = test_grouper(self.dataset_raw.test)
            self.scaler.fit(torch.tensor(self.train_set[0]["target"].transpose(1, 0)))
            self.global_mean = torch.mean(torch.tensor(self.train_set[0]["target"]), dim=-1)
        else:
            self.target_dim = 1
            multivariate = False
            self.num_test_dates = 1
            self.train_set = self.dataset_raw.train
            self.test_set = self.dataset_raw.test
            self.test_set = self.truncate_test(self.test_set)
            self.global_mean = torch.mean(
                torch.tensor(self.train_set[0]["target"]), dim=-1
            )  # TODO: check this

        if split_val:
            self.train_set, self.val_set = self.split_train_val(self.train_set)
        else:
            self.val_set = self.test_set

        if multivariate:
            self.expected_ndim = 2
        else:
            self.expected_ndim = 1

    def __read_data(self, dataset, path):
        self.dataset_raw = datasets.get_dataset(
            dataset, path=Path(path), regenerate=False
        )

        # Meta parameters
        self.target_dim = int(self.dataset_raw.metadata.feat_static_cat[0].cardinality)
        self.freq = self.dataset_raw.metadata.freq.upper()
        self.lags_list = get_lags(self.freq)
        self.prediction_length = (
            self.dataset_raw.metadata.prediction_length
            if self.prediction_length is None
            else self.prediction_length
        )
        self.context_length = (
            self.dataset_raw.metadata.prediction_length * self.context_length_factor
            if self.context_length is None
            else self.context_length
        )
        self.history_length = (
            (self.context_length + max(self.lags_list))
            if self.history_length is None
            else self.history_length
        )

    def get_iter_dataset(self, mode: str):
        if mode == "train":
            return super().get_iter_dataset(self.train_set, mode)
        elif mode == "val":
            return super().get_iter_dataset(self.val_set, mode)
        elif mode == "test":
            return super().get_iter_dataset(self.test_set, mode)

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
                trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][
                    :trunc_train_len
                ]
                univariate = True
            elif len(train_seq[FieldName.TARGET].shape) == 2:
                trunc_train_len = train_seq[FieldName.TARGET].shape[1] - offset
                trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][
                    :, :trunc_train_len
                ]
            else:
                raise ValueError(
                    f"Invalid Data Shape: {str(len(train_seq[FieldName.TARGET].shape))}"
                )

            trunc_train_list.append(trunc_train_seq)

            # construct val set by rolling
            for i in range(self.num_test_dates):
                val_seq = deepcopy(train_seq)
                rolling_len = trunc_train_len + self.prediction_length * (i + 1)
                if univariate:
                    val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][
                        trunc_train_len
                        + self.prediction_length * (i - 1)
                        - self.context_length : rolling_len
                    ]
                else:
                    val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][
                        :, :rolling_len
                    ]

                val_set_list.append(val_seq)

        trunc_train_set = ListDataset(
            trunc_train_list, freq=self.freq, one_dim_target=univariate
        )

        val_set = ListDataset(val_set_list, freq=self.freq, one_dim_target=univariate)

        return trunc_train_set, val_set

    def truncate_test(self, test_set):
        trunc_test_list = []
        for test_seq in iter(test_set):
            # truncate train set
            trunc_test_seq = deepcopy(test_seq)

            trunc_test_seq[FieldName.TARGET] = trunc_test_seq[FieldName.TARGET][
                -(self.prediction_length * 2 + self.context_length) :
            ]

            trunc_test_list.append(trunc_test_seq)

        trunc_test_set = ListDataset(
            trunc_test_list, freq=self.freq, one_dim_target=True
        )

        return trunc_test_set
