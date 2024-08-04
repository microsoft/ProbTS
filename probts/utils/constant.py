DATA_TO_FORECASTER_ARGS = (
    "target_dim", # list if multi-dataset
    # "history_length", # not sure yet
    "context_length", # set to max
    "prediction_length", # set to max of list
    "lags_list", # list if multi-dataset (get from freq)
    "freq", # list if multi-dataset (different freq for each dataset)
    "time_feat_dim", # int
    # "global_mean", # not sure yet
    "dataset",
)

DATA_TO_MODEL_ARGS = (
    "scaler", # list if multi-dataset
)

LIST_ARGS_PRETRAIN = (
    "scaler",
    "target_dim",
    "freq",
    "dataset",
    "lags_list",
    "prediction_length",
)

PROBTS_DATA_KEYS = [
    "target_dimension_indicator",
    "past_time_feat",
    "past_target_cdf",
    "past_observed_values",
    "past_is_pad",
    "future_time_feat",
    "future_target_cdf",
    "future_observed_values",
]
