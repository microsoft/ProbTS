DATA_TO_FORECASTER_ARGS = [
    "target_dim",
    "history_length",
    "context_length",
    "prediction_length",
    "lags_list",
    "freq",
    "time_feat_dim",
    "global_mean",
    "dataset",
]

DATA_TO_MODEL_ARGS = [
    "scaler",
]

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
