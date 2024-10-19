# ---------------------------------------------------------------------------------
# Portions of this file are derived from GluonTS
# - Source: https://github.com/awslabs/gluonts
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

from torch.utils.data import IterableDataset
from gluonts.env import env
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    SelectFields,
    Transformation,
    Chain,
    ValidationSplitSampler,
    ExpectedNumInstanceSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
    InstanceSplitter
)
from .time_features import fourier_time_features_from_frequency, AddCustomizedTimeFeatures
from .stsf_datasets import TransformedIterableDataset
from typing import Union
from gluonts.dataset.common import DataEntry
from gluonts.transform import InstanceSampler
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
from gluonts.zebras._util import pad_axis
from gluonts.dataset.common import DataEntry
import random
from gluonts.transform._base import FlatMapTransformation
import sys

class MultiHorizonDataset():

    def __init__(
        self,
        input_names: list,
        context_length: Union[int, list],
        prediction_length: Union[int, list],
        freq: str,
        ctx_range: Union[int, list]=None,
        pred_range: Union[int, list]=None,
        val_ctx_range: Union[int, list]=None,
        val_pred_range: Union[int, list]=None,
        multivariate: bool = True,
        continous_sample: bool = False,
    ):
        super().__init__()
        self.input_names_ = input_names
        self.ctx_range = ctx_range
        self.pred_range = pred_range
        self.val_ctx_range = val_ctx_range
        self.val_pred_range = val_pred_range
        self.context_length = context_length
        self.prediction_length=prediction_length
        self.continous_sample = continous_sample
        
        self.freq = freq
        if multivariate:
            self.expected_ndim = 2
        else:
            self.expected_ndim = 1

    def get_sampler(self):
        # for training
        train_min_past = min(self.ctx_range)
        train_min_future = min(self.pred_range)
        
        # for validation
        val_min_past = max(self.val_ctx_range)
        val_min_future = max(self.val_pred_range)
        
        # for testing
        if (type(self.context_length).__name__=='list'):
            test_min_past = max(self.context_length)
        else:
            test_min_past=self.context_length
        
        if (type(self.pred_range).__name__=='list'):
            test_min_future = max(self.prediction_length)
        else:
            test_min_future=self.prediction_length

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=train_min_past,
            min_future=train_min_future,
        )

        self.val_sampler = ValidationSplitSampler(
            min_past=val_min_past,
            min_future=val_min_future,
        )
        
        self.test_sampler = ValidationSplitSampler(
            min_past=test_min_past,
            min_future=test_min_future,
        )

        
    def create_transformation(self, data_stamp=None, pred_len=None) -> Transformation:
        if data_stamp is None:
            if self.freq in ["M", "W", "D", "B", "H", "min", "T"]:
                time_features = fourier_time_features_from_frequency(self.freq)
            else:
                time_features = fourier_time_features_from_frequency('D')
            self.time_feat_dim = len(time_features) * 2
            time_feature_func = AddTimeFeatures
        else:
            self.time_feat_dim = data_stamp.shape[-1]
            time_features = data_stamp
            time_feature_func = AddCustomizedTimeFeatures
            
        if pred_len is None:
            pred_len = max(self.pred_range)
        else:
            pred_len = max(pred_len)
            
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=self.expected_ndim,
                ),
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                time_feature_func(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features,
                    pred_length=pred_len,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str, pred_len=None):
        assert mode in ["train", "val", "test"]

        self.get_sampler()
        instance_sampler = {
            "train": self.train_sampler,
            "val": self.val_sampler,
            "test": self.test_sampler,
        }[mode]

        if mode == "train":
            past_length = self.ctx_range
            future_length = self.pred_range
        elif mode == 'val':
            past_length = self.val_ctx_range
            if pred_len is None:
                future_length = self.val_pred_range
            else:
                future_length = pred_len
        else:
            past_length = self.context_length
            if pred_len is None:
                future_length = self.prediction_length
            else:
                future_length = pred_len
            
            
        return MultiHorizonSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=future_length,
            mode=mode,
            continous_sample=self.continous_sample,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )


    def get_iter_dataset(self, dataset: Dataset, mode: str, data_stamp=None, pred_len=None) -> IterableDataset:
        assert mode in ["train", "val", "test"]

        transform = self.create_transformation(data_stamp, pred_len=pred_len)
            
            
        if mode == 'train':
            with env._let(max_idle_transforms=100):
                instance_splitter = self.create_instance_splitter(mode)
        else:
            instance_splitter = self.create_instance_splitter(mode, pred_len=pred_len)


        input_names = self.input_names_

        iter_dataset = TransformedIterableDataset(
            dataset,
            transform=transform
            + instance_splitter
            + SelectFields(input_names),
            is_train=True if mode == 'train' else False
        )

        return iter_dataset


class MultiHorizonSplitter(FlatMapTransformation):
    """
    Split instances from a dataset, by slicing the target and other time series
    fields at points in time selected by the specified sampler. The assumption
    is that all time series fields start at the same time point.

    It is assumed that time axis is always the last axis.

    The ``target_field`` and each field in ``time_series_fields`` are removed and
    replaced by two new fields, with prefix `past_` and `future_` respectively.

    A ``past_is_pad`` is also added, that indicates whether values at a given
    time point are padding or not.

    Parameters
    ----------

    target_field
        field containing the target
    is_pad_field
        output field indicating whether padding happened
    start_field
        field containing the start date of the time series
    forecast_start_field
        output field that will contain the time point where the forecast starts
    instance_sampler
        instance sampler that provides sampling indices given a time series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    lead_time
        gap between the past and future windows (default: 0)
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout (default: True)
    time_series_fields
        fields that contains time series, they are split in the same interval
        as the target (default: None)
    dummy_value
        Value to use for padding. (default: 0.0)
    """

    # @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        past_length: Union[int, list],
        future_length: Union[int, list],
        mode: str,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        dummy_value: float = 0.0,
        continous_sample: bool = False,
    ) -> None:
        super().__init__()

        # assert future_length > 0, "The value of `future_length` should be > 0"

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.continous_sample = continous_sample
        
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value
        self.mode = mode

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def _split_array(
        self, array: np.ndarray, idx: int, past_length: int, future_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if idx >= past_length:
            past_piece = array[..., idx - past_length : idx]
        else:
            past_piece = pad_axis(
                array[..., :idx],
                axis=-1,
                left=past_length - idx,
                value=self.dummy_value,
            )

        future_start = idx + self.lead_time
        future_slice = slice(future_start, future_start + future_length)
        future_piece = array[..., future_slice]

        return past_piece, future_piece

    def _split_instance(self, entry: DataEntry, idx: int, is_train) -> DataEntry:
        slice_cols = self.ts_fields + [self.target_field]
        dtype = entry[self.target_field].dtype
        entry = entry.copy()
        
        if is_train:
            if self.continous_sample:
                past_len = random.randint(min(self.past_length), max(self.past_length))
                pred_len = random.randint(min(self.future_length), max(self.future_length))
            else:
                past_len = random.choice(self.past_length) 
                pred_len = random.choice(self.future_length) 
        else:
            past_len = max(self.past_length)
            pred_len = max(self.future_length)

        for ts_field in slice_cols:
            past_piece, future_piece = self._split_array(entry[ts_field], idx, past_length=past_len, future_length=pred_len)

            if self.output_NTC:
                past_piece = past_piece.transpose()
                future_piece = future_piece.transpose()

            entry[self._past(ts_field)] = past_piece
            entry[self._future(ts_field)] = future_piece
            del entry[ts_field]

        pad_indicator = np.zeros(past_len, dtype=dtype)
        pad_length = max(past_len - idx, 0)
        pad_indicator[:pad_length] = 1

        entry[self._past(self.is_pad_field)] = pad_indicator
        entry[self.forecast_start_field] = (
            entry[self.start_field] + idx + self.lead_time
        )
        entry['context_length'] = past_len
        entry['prediction_length'] = pred_len

        return entry

    def flatmap_transform(
            self, entry: DataEntry, is_train: bool
        ) -> Iterator[DataEntry]:
        sampled_indices = self.instance_sampler(entry[self.target_field])
        
        for idx in sampled_indices:
            yield self._split_instance(entry, idx, is_train)