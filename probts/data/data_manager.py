from typing import Union

import torch
from gluonts.dataset.repository import dataset_names

from probts.utils.constant import PROBTS_DATA_KEYS, DATA_TO_FORECASTER_ARGS, DATA_TO_MODEL_ARGS, LIST_ARGS_PRETRAIN

from .ltsf_datasets import LongTermTSDatasetLoader
from .stsf_datasets import GluonTSDatasetLoader
from .probts_datasets import MultiIterableDataset


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
        dataset: Union[str, list[str]],
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
        is_pretrain: bool = False,
    ):
        self.dataset = dataset
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
            is_pretrain,
        ]
        short_term_specific_args = {"context_length_factor": context_length_factor}
        long_term_specific_args = {"timeenc": timeenc}

        if isinstance(dataset, str) or len(dataset) == 1:
            dataset = dataset if isinstance(dataset, str) else dataset[0]
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

            for key in DATA_TO_FORECASTER_ARGS + DATA_TO_MODEL_ARGS:
                if key not in self.__dict__ or self.__dict__[key] is None:
                    assert key in probts_dataset.__dict__, f"{key} not in probts_dataset"
                    assert probts_dataset.__dict__[key] is not None, f"{key} is None"
                    setattr(self, key, getattr(probts_dataset, key))

            print(
                f"context_length: {self.context_length}, prediction_length: {self.prediction_length}"
            )
            if scaler == "standard":
                print(f"variate-specific normalization: {var_specific_norm}")
        else:  # Load multiple datasets
            dataset_list = dataset
            probts_dataset_list = []
            for dataset in dataset_list:
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
            self.train_iter_dataset = self.__get_iter_multi_dataset(mode="train")
            self.val_iter_dataset = self.__get_iter_multi_dataset(mode="val")
            self.test_iter_dataset = self.__get_iter_multi_dataset(mode="test")

            for key in DATA_TO_FORECASTER_ARGS + DATA_TO_MODEL_ARGS:
                if key not in self.__dict__ or self.__dict__[key] is None:
                    for probts_dataset in probts_dataset_list:
                        assert key in probts_dataset.__dict__, f"{key} not in probts_dataset"
                        assert probts_dataset.__dict__[key] is not None, f"{key} is None"
                    if key in LIST_ARGS_PRETRAIN:
                        setattr(self, key, [getattr(probts_dataset, key) for probts_dataset in probts_dataset_list])
                    else: # context_length, prediction_length, time_feat_dim
                        setattr(self, key, max([getattr(probts_dataset, key) for probts_dataset in probts_dataset_list]))

            self.target_dim = 1

    def __get_iter_multi_dataset(self, mode):
        return MultiIterableDataset(self.probts_dataset_list, mode)

    def __str__(self):
        dataset_str = '-'.join(self.dataset) if isinstance(self.dataset, list) else self.dataset
        return f"data_{dataset_str}_ctx_{self.context_length}_pred_{self.prediction_length}"