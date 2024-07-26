from dataclasses import dataclass, field
from typing import List

import numpy as np
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.env import env
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
)
from torch.utils.data import IterableDataset

from probts.utils import IdentityScaler, StandardScaler, TemporalScaler
from probts.utils.constant import PROBTS_DATA_KEYS

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
        self.n_samples = 0
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
        self.probabilities = np.array(
            [1 / len(probts_dataset_list)] * len(probts_dataset_list)
        )
        self.iterators = [iter(iterable) for iterable in self.iterables]

    def __iter__(self):
        return self

    def __next__(self):
        idx = np.random.choice(len(self.iterators), p=self.probabilities)
        self.n_samples += 1
        print("current idx: ", idx, "total samples: ", self.n_samples)
        try:
            return next(self.iterators[idx])
        except StopIteration:
            self.iterators[idx] = iter(self.iterables[idx])
            return next(self.iterators[idx])


@dataclass
class ProbTSDataset:
    dataset: str
    path: str
    context_length: int
    history_length: int
    prediction_length: int
    scaler: str = "none"
    var_specific_norm: bool = (True,)
    test_rolling_length: int = (96,)
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

    def __get_sampler(self):
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_instances=1,
            min_past=self.history_length,
            min_future=self.prediction_length,
        )

        self.val_sampler = ValidationSplitSampler(
            min_past=self.history_length,
            min_future=self.prediction_length,
        )

        self.test_sampler = ValidationSplitSampler(
            min_past=self.history_length,
            min_future=self.prediction_length,
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
                    pred_length=self.prediction_length,
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
            ]
        )

    def __create_instance_splitter(self, mode: str):
        assert mode in ["train", "val", "test"]

        self.__get_sampler()
        instance_sampler = {
            "train": self.train_sampler,
            "val": self.val_sampler,
            "test": self.test_sampler,
        }[mode]

        return InstanceSplitter(
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

    def get_iter_dataset(
        self, dataset: Dataset, mode: str, data_stamp=None
    ) -> IterableDataset:
        assert mode in ["train", "val", "test"]

        transform = self.__create_transformation(data_stamp)
        if mode == "train":
            with env._let(max_idle_transforms=100):
                instance_splitter = self.__create_instance_splitter(mode)
        else:
            instance_splitter = self.__create_instance_splitter(mode)

        input_names = self.input_names_

        rename_fields = RenameFields(
            {
                f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
            }
        )

        if self.is_pretrain:
            sample_target_dim = SampleTargetDim(
                field_name="target_dimension_indicator",
                target_field=FieldName.TARGET,
                observed_values_field=FieldName.OBSERVED_VALUES,
                num_samples=1,  # univaraite time-series in pretraining
                shuffle=True,
            )
            all_transforms = (
                transform
                + instance_splitter
                + sample_target_dim
                + rename_fields
                + SelectFields(input_names)
            )
        else:
            all_transforms = (
                transform
                + instance_splitter
                + rename_fields
                + SelectFields(input_names)
            )

        iter_dataset = TransformedIterableDataset(
            dataset,
            transform=all_transforms,
            is_train=True if mode == "train" else False,
        )

        return iter_dataset
