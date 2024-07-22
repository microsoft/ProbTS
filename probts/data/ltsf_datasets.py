from copy import deepcopy

import torch
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from .probts_datasets import ProbTSDataset
from .time_features import get_lags
from .utils import get_LTSF_borders, get_LTSF_Dataset, get_LTSF_info


class LongTermTSDatasetLoader(ProbTSDataset):
    def __init__(
        self,
        input_names: list,
        context_length: int,
        history_length: int,
        prediction_length: int,
        dataset: str,
        path: str,
        scaler: str,
        var_specific_norm: bool,
        **kwargs,
    ):
        super().__init__(
            input_names,
            context_length,
            history_length,
            prediction_length,
            scaler,
            var_specific_norm,
        )

        self.timeenc = kwargs.get("timeenc", 1)
        self.__read_data(dataset, path)

        train_data = self.dataset_raw[: self.border_end[0]]
        val_data = self.dataset_raw[: self.border_end[1]]
        test_data = self.dataset_raw[: self.border_end[2]]

        self.scaler.fit(torch.tensor(train_data.values))

        train_set = self.df_to_mvds(train_data, freq=self.freq)
        val_set = self.df_to_mvds(val_data, freq=self.freq)
        test_set = self.df_to_mvds(test_data, freq=self.freq)

        train_grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        test_grouper = MultivariateGrouper(max_target_dim=self.target_dim)

        self.group_train_set = train_grouper(train_set)
        self.group_val_set = test_grouper(val_set)
        self.group_test_set = test_grouper(test_set)

        self.group_val_set = self.get_rolling_test(
            self.group_val_set,
            self.border_begin[1],
            self.border_end[1],
            rolling_length=self.test_rolling_length,
        )
        self.group_test_set = self.get_rolling_test(
            self.group_test_set,
            self.border_begin[2],
            self.border_end[2],
            rolling_length=self.test_rolling_length,
        )

        self.global_mean = torch.mean(
            torch.tensor(self.group_train_set[0]["target"]), dim=-1
        )

    def __read_data(self, dataset, path):
        data_path, self.freq = get_LTSF_info(dataset)
        self.dataset_raw, self.data_stamp, self.target_dim, data_size = (
            get_LTSF_Dataset(path, data_path, timeenc=self.timeenc)
        )
        self.border_begin, self.border_end = get_LTSF_borders(dataset, data_size)

        assert data_size >= self.border_end[2], print(
            "\n The end index larger then data size!"
        )

        # Meta parameters
        self.lags_list = get_lags(self.freq)
        self.history_length = (
            (self.context_length + max(self.lags_list))
            if self.history_length is None
            else self.history_length
        )

    def get_iter_dataset(self, mode: str):
        if mode == "train":
            return super().get_iter_dataset(
                self.group_train_set, mode, self.data_stamp[: self.border_end[0]]
            )
        elif mode == "val":
            return super().get_iter_dataset(
                self.group_val_set, mode, self.data_stamp[: self.border_end[1]]
            )
        elif mode == "test":
            return super().get_iter_dataset(
                self.group_test_set, mode, self.data_stamp[: self.border_end[2]]
            )

    def df_to_mvds(self, df, freq="H"):
        datasets = []
        for variable in df.keys():
            ds = {
                "item_id": variable,
                "target": df[variable],
                "start": str(df.index[0]),
            }
            datasets.append(ds)
        dataset = ListDataset(datasets, freq=freq)
        return dataset

    def get_rolling_test(
        self, test_set, border_begin_idx, border_end_idx, rolling_length
    ):
        if (border_end_idx - border_begin_idx - self.prediction_length) < 0:
            raise ValueError(
                "The time steps in validation / testing set is less than prediction length."
            )

        num_test_dates = (
            int(
                (
                    (border_end_idx - border_begin_idx - self.prediction_length)
                    / rolling_length
                )
            )
            + 1
        )
        print("num_test_dates: ", num_test_dates)

        test_set = next(iter(test_set))
        rolling_test_seq_list = list()
        for i in range(num_test_dates):
            rolling_test_seq = deepcopy(test_set)
            rolling_end = border_begin_idx + self.prediction_length + i * rolling_length
            rolling_test_seq[FieldName.TARGET] = rolling_test_seq[FieldName.TARGET][
                :, :rolling_end
            ]
            rolling_test_seq_list.append(rolling_test_seq)

        rolling_test_set = ListDataset(
            rolling_test_seq_list, freq=self.freq, one_dim_target=False
        )
        return rolling_test_set
