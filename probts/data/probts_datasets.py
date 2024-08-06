from dataclasses import dataclass, field
from typing import List, Union
import itertools

import numpy as np
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RenameFields,
    SelectFields,
    SetFieldIfNotPresent,
    TargetDimIndicator,
    Transformation,
    TransformedDataset,
    ValidationSplitSampler,
    SampleTargetDim,
    Identity,
)
from torch.utils.data import IterableDataset

from probts.utils import IdentityScaler, StandardScaler, TemporalScaler
from probts.utils.constant import PROBTS_DATA_KEYS
from probts.data.multi_horizon_sampler import MultiHorizonSplitter

from .time_features import (
    AddCustomizedTimeFeatures,
    fourier_time_features_from_frequency,
)


class TransformedIterableDataset(IterableDataset):
    def __init__(
        self, dataset: Dataset, transform: Transformation, is_train: bool = True
    ):
        super().__init__()
        self.transformed_dataset = TransformedDataset(
            Cyclic(dataset) if is_train else dataset,
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        return iter(self.transformed_dataset)


class MultiIterableDataset(IterableDataset):
    def __init__(self, probts_dataset_list, mode):
        super().__init__()
        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode should be 'train', 'val', or 'test'."
        self.probts_dataset_list = probts_dataset_list
        self.mode = mode
        self.iterables = [
            dataset.get_iter_dataset(mode=mode) for dataset in probts_dataset_list
        ]
        self.iterables = list(itertools.chain(*self.iterables))
        self.iterators = [iter(iterable) for iterable in self.iterables]
        self.probabilities = np.array([1 / len(self.iterables)] * len(self.iterables))

    def __iter__(self):
        return self

    def __next__(self):
        idx = np.random.choice(len(self.iterators), p=self.probabilities)
        try:
            data = next(self.iterators[idx])
        except StopIteration:
            if self.mode == "train":
                self.iterators[idx] = iter(self.iterables[idx])
                data = next(self.iterators[idx])
            else:
                raise StopIteration
        data["dataset_idx"] = idx
        return data


@dataclass
class ProbTSDataset:
    dataset: str
    path: str
    context_length: int
    history_length: int
    prediction_length: Union[int, List[int], List[List[int]]]
    scaler: str = "none"
    var_specific_norm: bool = True
    test_rolling_length: int = 96
    is_pretrain: bool = False

    input_names_: List[str] = field(
        default_factory=lambda: PROBTS_DATA_KEYS, init=False
    )
    expected_ndim: int = field(default=2, init=False)

    def __post_init__(self):
        if self.scaler == "standard":
            self.scaler = StandardScaler(var_specific=self.var_specific_norm)
        elif self.scaler == "temporal":
            self.scaler = TemporalScaler()
        else:
            self.scaler = IdentityScaler()

        # handle multiple prediction lengths
        self.prediction_length_list = None
        if isinstance(self.prediction_length, list):  # list of int, [96, 192]
            self.prediction_length_list = sorted(self.prediction_length)
            self.max_prediction_length = int(max(self.prediction_length))
        else:
            self.max_prediction_length = int(self.prediction_length)

    def __get_sampler(self):
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_instances=1,
            min_past=self.history_length,
            min_future=self.max_prediction_length,
        )

        self.val_sampler = ValidationSplitSampler(
            min_past=self.history_length,
            min_future=self.max_prediction_length,
        )

        self.test_sampler = ValidationSplitSampler(
            min_past=self.history_length,
            min_future=self.max_prediction_length,
        )

    def __create_transformation(self, data_stamp=None) -> Transformation:
        if data_stamp is None:
            if self.freq in ["M", "W", "D", "B", "H", "min", "T"]:
                time_features = fourier_time_features_from_frequency(self.freq)
            else:
                time_features = fourier_time_features_from_frequency("D")
            self.time_feat_dim = len(time_features) * 2
            time_feature_func = AddTimeFeatures
        else:
            self.time_feat_dim = data_stamp.shape[-1]
            time_features = data_stamp
            time_feature_func = AddCustomizedTimeFeatures

        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                time_feature_func(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features,
                    pred_length=self.max_prediction_length,
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
            ]
        )

    def __create_instance_splitter(self, mode: str, pred_len: int):
        assert mode in ["train", "val", "test"]

        self.__get_sampler()
        instance_sampler = {
            "train": self.train_sampler,
            "val": self.val_sampler,
            "test": self.test_sampler,
        }[mode]

        # if False:
        if self.is_pretrain:
            if self.prediction_length_list is not None and mode == "train":
                instance_splitter = MultiHorizonSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    instance_sampler=instance_sampler,
                    past_length=[self.history_length],
                    future_length=self.prediction_length_list,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                )
            else:
                instance_splitter = InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    instance_sampler=instance_sampler,
                    past_length=self.history_length,
                    future_length=pred_len,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                )
        else:
            instance_splitter = InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self.history_length,
                future_length=self.prediction_length,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                ],
            )
        return instance_splitter

    def get_iter_dataset(
        self, dataset: Dataset, mode: str, data_stamp=None
    ) -> IterableDataset:
        assert mode in ["train", "val", "test"]

        if self.is_pretrain:
            sample_target_dim = SampleTargetDim(
                field_name="target_dimension_indicator",
                target_field=FieldName.TARGET,
                observed_values_field=FieldName.OBSERVED_VALUES,
                num_samples=1,  # univaraite time-series in pretraining
                shuffle=True,
            )
        else:
            sample_target_dim = Identity()

        rename_fields = RenameFields(
            {
                f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
            }
        )

        if self.prediction_length_list is not None:
            assert isinstance(dataset, list)
            iter_dataset_list = []
            for i, pred_len in enumerate(self.prediction_length_list):
                all_transforms = (
                    self.__create_transformation(data_stamp)
                    + self.__create_instance_splitter(mode, pred_len)
                    + sample_target_dim
                    + rename_fields
                    + SelectFields(self.input_names_)
                )

                iter_dataset = TransformedIterableDataset(
                    dataset if mode == "train" else dataset[i],
                    transform=all_transforms,
                    is_train=(mode == "train"),
                )
                iter_dataset_list.append(iter_dataset)
            return iter_dataset_list

        else:
            all_transforms = (
                self.__create_transformation(data_stamp)
                + self.__create_instance_splitter(mode, self.prediction_length)
                + sample_target_dim
                + rename_fields
                + SelectFields(self.input_names_)
            )

            iter_dataset = TransformedIterableDataset(
                dataset,
                transform=all_transforms,
                is_train=True if mode == "train" else False,
            )

            return iter_dataset
