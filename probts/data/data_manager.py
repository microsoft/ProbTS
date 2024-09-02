from typing import Union, List

from gluonts.dataset.repository import dataset_names

from probts.utils.constant import (
    DATA_TO_FORECASTER_ARGS,
    DATA_TO_MODEL_ARGS,
    LIST_ARGS_PRETRAIN,
)

from probts.utils import get_unique_list
from .ltsf_datasets import LongTermTSDatasetLoader
from .stsf_datasets import GluonTSDatasetLoader
from .probts_datasets import MultiIterableDataset


class DataManager:
    def __init__(
        self,
        dataset: Union[str, list[str]],
        split_val: bool = True,
        test_sampling: str = "ctx",  # ['arrange', 'ctx', 'pred']
        # **kwargs
        path: str = "./datasets",
        history_length: int = None,
        context_length: int = None,
        prediction_length: Union[int, List[int], List[List[int]]] = None,
        scaler: str = "none",
        var_specific_norm: bool = True,
        test_rolling_length: int = 96,
        is_pretrain: bool = False,
        context_length_factor: int = 1,
        timeenc: int = 1,
        data_path: str = None,
        freq: str = None,
    ):
        self.dataset = dataset
        # self.test_rolling_length = test_rolling_length
        self.global_mean = None
        # self.split_val = split_val
        # self.test_sampling = test_sampling
        self.prediction_length_raw = prediction_length

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
        long_term_specific_args = {"timeenc": timeenc, "data_path": data_path, "freq": freq}

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
                    assert (
                        key in probts_dataset.__dict__
                    ), f"{key} not in probts_dataset"
                    assert probts_dataset.__dict__[key] is not None, f"{key} is None"
                    setattr(self, key, getattr(probts_dataset, key))

            print(
                f"context_length: {self.context_length}, prediction_length: {self.prediction_length}"
            )
            if scaler == "standard":
                print(f"variate-specific normalization: {var_specific_norm}")
            self.dataloader_id_mapper = None

        else:  # Load multiple datasets
            dataset_list = dataset
            probts_dataset_list = []
            for data_idx, dataset in enumerate(dataset_list):
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

                probts_dataset_args_i = [
                    args[data_idx] if isinstance(args, list) else args
                    for args in probts_dataset_args
                ]
                probts_dataset = dataset_class(
                    dataset,
                    *probts_dataset_args_i,
                    **specific_args,
                )
                probts_dataset_list.append(probts_dataset)

            self.probts_dataset_list = probts_dataset_list
            self.train_iter_dataset = self.__get_iter_multi_dataset(mode="train")
            self.val_iter_dataset = self.__get_iter_multi_dataset(mode="val")
            self.test_iter_dataset = self.__get_iter_multi_dataset(mode="test")

            for key in DATA_TO_FORECASTER_ARGS + DATA_TO_MODEL_ARGS:
                for probts_dataset in probts_dataset_list:
                    assert (
                        key in probts_dataset.__dict__
                    ), f"{key} not in probts_dataset"
                    assert probts_dataset.__dict__[key] is not None, f"{key} is None"
                if key in LIST_ARGS_PRETRAIN:
                    setattr(
                        self,
                        key,
                        [
                            getattr(probts_dataset, key)
                            for probts_dataset in probts_dataset_list
                        ],
                    )
                else:  # context_length, prediction_length, time_feat_dim
                    setattr(
                        self,
                        key,
                        max(
                            [
                                getattr(probts_dataset, key)
                                for probts_dataset in probts_dataset_list
                            ]
                        ),
                    )
            
            self.prediction_length = sorted(get_unique_list(self.prediction_length))
            self.dataloader_id_mapper = self.__create_mapper(prediction_length)
            self.target_dim = 1

    def __get_iter_multi_dataset(self, mode):
        assert isinstance(
            self.dataset, list
        ), "dataset should be a list if loading multiple datasets"
        if mode == "train":
            return MultiIterableDataset(self.probts_dataset_list, mode)
        else:
            multi_dataset = {}
            for i, dataset in enumerate(self.dataset):
                iter_dataset = self.probts_dataset_list[i].get_iter_dataset(mode)
                if isinstance(iter_dataset, list):
                    assert len(iter_dataset) == len(self.prediction_length_raw[i])
                    for iter_dataset_i, pred_len in zip(
                        iter_dataset, self.prediction_length_raw[i]
                    ):
                        multi_dataset[f"{dataset}_pred_{pred_len}"] = iter_dataset_i
                else:
                    multi_dataset[dataset] = iter_dataset
            return multi_dataset

    @staticmethod
    def __create_mapper(lst: list):
        mapping = []
        
        for i, sublist in enumerate(lst):
            for _ in sublist:
                mapping.append(i)
        
        def mapper(index):
            if index < len(mapping) and index >= 0:
                return mapping[index]
            else:
                raise IndexError("Dataloader index out of range")
    
        return mapper

    def __str__(self):
        dataset_str = (
            "-".join(self.dataset) if isinstance(self.dataset, list) else self.dataset
        )
        return f"data_{dataset_str}_ctx_{self.context_length}_pred_{self.prediction_length}"
