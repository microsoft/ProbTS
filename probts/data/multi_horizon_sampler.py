import random
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from gluonts.dataset.common import DataEntry
from gluonts.transform import InstanceSampler
from gluonts.transform._base import FlatMapTransformation
from gluonts.zebras._util import pad_axis


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
        past_length: list,
        future_length: list,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: List[str] = [],
        dummy_value: float = 0.0,
        continous_sample: bool = False,
        curriculum: bool = False,
    ) -> None:
        super().__init__()

        # assert future_length > 0, "The value of `future_length` should be > 0"

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.continous_sample = continous_sample
        self.curriculum = curriculum
        
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value

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
            if self.curriculum:
                past_len = max(self.past_length)
                pred_len = max(self.future_length)
            elif self.continous_sample:
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