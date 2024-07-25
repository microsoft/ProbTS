from typing import Union

import numpy as np
import torch
from gluonts.dataset.repository import dataset_names

from probts.utils.constant import PROBTS_DATA_KEYS

from .ltsf_datasets import LongTermTSDatasetLoader
from .stsf_datasets import GluonTSDatasetLoader


class ProbTSBatchData:
    input_names_ = PROBTS_DATA_KEYS

    def __init__(self, data_dict, device):
        self.__dict__.update(data_dict)
        if len(self.__dict__["past_target_cdf"].shape) == 2:
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
        self.__dict__["target_dimension_indicator"] = self.__dict__[
            "target_dimension_indicator"
        ][:, :1]
        for input in [
            "past_target_cdf",
            "past_observed_values",
            "future_target_cdf",
            "future_observed_values",
        ]:
            self.__dict__[input] = self.__dict__[input].unsqueeze(-1)

    def process_pad(self):
        if self.__dict__["past_is_pad"] is not None:
            self.__dict__["past_observed_values"] = torch.min(
                self.__dict__["past_observed_values"],
                1 - self.__dict__["past_is_pad"].unsqueeze(-1),
            )


class DataManager:
    def __init__(
        self,
        datasets: Union[str, list[str]],
        path: str = "./datasets",
        history_length: int = None,
        context_length: int = None,
        prediction_length: int = None,
        test_rolling_length: int = 96,
        split_val: bool = True,
        scaler: str = "none",
        test_sampling: str = "ctx",  # ['arrange', 'ctx', 'pred']
        context_length_factor: int = 1,
        timeenc: int = 1,
        var_specific_norm: bool = True,
    ):
        self.dataset = datasets
        # self.test_rolling_length = test_rolling_length
        self.global_mean = None
        # self.split_val = split_val
        # self.test_sampling = test_sampling

        probts_dataset_args = [
            path,
            context_length,
            history_length,
            prediction_length,
            scaler,
            var_specific_norm,
            test_rolling_length,
        ]
        short_term_specific_args = {"context_length_factor": context_length_factor}
        long_term_specific_args = {"timeenc": timeenc}

        if isinstance(datasets, str):
            dataset = datasets
            if dataset in dataset_names:  # Use gluonts to load short-term datasets.
                print("Loading Short-term Datasets: {dataset}".format(dataset=dataset))
                dataset_class = GluonTSDatasetLoader
                specific_args = short_term_specific_args

            else:  # Load long-term datasets.
                print("Loading Long-term Datasets: {dataset}".format(dataset=dataset))
                dataset_class = LongTermTSDatasetLoader
                specific_args = long_term_specific_args

            probts_dataset = dataset_class(
                dataset,
                *probts_dataset_args,
                **specific_args,
            )
            self.train_iter_dataset = probts_dataset.get_iter_dataset(mode="train")
            self.val_iter_dataset = probts_dataset.get_iter_dataset(mode="val")
            self.test_iter_dataset = probts_dataset.get_iter_dataset(mode="test")

            self.global_mean = probts_dataset.global_mean
            self.time_feat_dim = probts_dataset.time_feat_dim
            self.freq = probts_dataset.freq
            self.context_length = probts_dataset.context_length
            self.history_length = probts_dataset.history_length
            self.prediction_length = probts_dataset.prediction_length
            self.lags_list = probts_dataset.lags_list
            self.target_dim = probts_dataset.target_dim
            self.scaler = probts_dataset.scaler

            print(
                f"context_length: {self.context_length}, prediction_length: {self.prediction_length}"
            )
            if scaler == "standard":
                print(f"variate-specific normalization: {var_specific_norm}")
        else:  # Load multiple datasets
            probts_dataset_list = []
            for dataset in datasets:
                if dataset in dataset_names:  # Use gluonts to load short-term datasets.
                    print(
                        "Loading Short-term Datasets: {dataset}".format(dataset=dataset)
                    )
                    dataset_class = GluonTSDatasetLoader
                    specific_args = short_term_specific_args

                else:  # Load long-term datasets.
                    print(
                        "Loading Long-term Datasets: {dataset}".format(dataset=dataset)
                    )
                    dataset_class = LongTermTSDatasetLoader
                    specific_args = long_term_specific_args

                probts_dataset = dataset_class(
                    dataset,
                    *probts_dataset_args,
                    **specific_args,
                )
                probts_dataset_list.append(probts_dataset)
            self.probts_dataset_list = probts_dataset_list
            self.train_iter_dataset = self.get_iter_multi_dataset(mode="train")
            self.val_iter_dataset = self.get_iter_multi_dataset(mode="val")
            self.test_iter_dataset = self.get_iter_multi_dataset(mode="test")

    def get_iter_multi_dataset(self, mode):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode should be 'train', 'val', or 'test'."

        iterables = []
        for probts_dataset in self.probts_dataset_list:
            if mode == "train":
                iterables.append(probts_dataset.get_iter_dataset(mode="train"))
            elif mode == "val":
                iterables.append(probts_dataset.get_iter_dataset(mode="val"))
            elif mode == "test":
                iterables.append(probts_dataset.get_iter_dataset(mode="test"))

        probabilities = np.array(
            [1 / len(self.probts_dataset_list)] * len(self.probts_dataset_list)
        )
        print(probabilities)
        iterators = [iter(iterable) for iterable in iterables]

        while True:
            idx = np.random.choice(len(iterators), p=probabilities)
            print(idx, self.dataset[idx])
            try:
                yield next(iterators[idx])
            except StopIteration:
                iterators[idx] = iter(iterables[idx])
                yield next(iterators[idx])
