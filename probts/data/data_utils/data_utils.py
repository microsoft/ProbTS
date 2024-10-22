from copy import deepcopy
import math
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName


def split_train_val(train_set, num_test_dates, context_length, prediction_length, freq):
    trunc_train_list = []
    val_set_list = []
    univariate = False

    for train_seq in iter(train_set):
        # truncate train set
        offset = num_test_dates * prediction_length
        trunc_train_seq = deepcopy(train_seq)

        if len(train_seq[FieldName.TARGET].shape) == 1:
            trunc_train_len = train_seq[FieldName.TARGET].shape[0] - offset
            trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][:trunc_train_len]
            univariate = True
        elif len(train_seq[FieldName.TARGET].shape) == 2:
            trunc_train_len = train_seq[FieldName.TARGET].shape[1] - offset
            trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][:, :trunc_train_len]
        else:
            raise ValueError(f"Invalid Data Shape: {str(len(train_seq[FieldName.TARGET].shape))}")

        trunc_train_list.append(trunc_train_seq)

        # construct val set by rolling
        for i in range(num_test_dates):
            val_seq = deepcopy(train_seq)
            rolling_len = trunc_train_len + prediction_length * (i+1)
            if univariate:
                val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][trunc_train_len + prediction_length * (i-1) - context_length : rolling_len]
            else:
                val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][:, :rolling_len]
            
            val_set_list.append(val_seq)

    trunc_train_set = ListDataset(
        trunc_train_list, freq=freq, one_dim_target=univariate
    )

    val_set = ListDataset(
        val_set_list, freq=freq, one_dim_target=univariate
    )
    
    return trunc_train_set, val_set


def truncate_test(test_set, context_length, prediction_length, freq):
    trunc_test_list = []
    for test_seq in iter(test_set):
        # truncate train set
        trunc_test_seq = deepcopy(test_seq)

        trunc_test_seq[FieldName.TARGET] = trunc_test_seq[FieldName.TARGET][- (prediction_length * 2 + context_length):]

        trunc_test_list.append(trunc_test_seq)

    trunc_test_set = ListDataset(
        trunc_test_list, freq=freq, one_dim_target=True
    )

    return trunc_test_set


# def get_rolling_test(test_set, prediction_length, border_begin_idx, border_end_idx, rolling_length, freq):
#     if (border_end_idx - border_begin_idx - prediction_length) < 0:
#         raise ValueError("The time steps in validation / testing set is less than prediction length.")
    
#     num_test_dates = int(((border_end_idx - border_begin_idx - prediction_length) / rolling_length)) + 1
#     print("num_test_dates: ", num_test_dates)

#     test_set = next(iter(test_set))
#     rolling_test_seq_list = list()
#     for i in range(num_test_dates):
#         rolling_test_seq = deepcopy(test_set)
#         rolling_end = border_begin_idx + prediction_length + i * rolling_length
#         rolling_test_seq[FieldName.TARGET] = rolling_test_seq[FieldName.TARGET][:, :rolling_end]
#         rolling_test_seq_list.append(rolling_test_seq)

#     rolling_test_set = ListDataset(
#         rolling_test_seq_list, freq=freq, one_dim_target=False
#     )
#     return rolling_test_set


def get_rolling_test(stage, test_set, border_begin_idx, border_end_idx, rolling_length, pred_len, freq):
    num_test_dates = math.ceil(((border_end_idx - border_begin_idx - pred_len) / rolling_length))
    print(f"{stage}  pred_len: {pred_len} : num_test_dates: {num_test_dates}")

    test_set = next(iter(test_set))
    rolling_test_seq_list = list()
    for i in range(num_test_dates):
        rolling_test_seq = deepcopy(test_set)
        rolling_end = border_begin_idx + pred_len + i * rolling_length
        rolling_test_seq[FieldName.TARGET] = rolling_test_seq[FieldName.TARGET][:, :rolling_end]
        rolling_test_seq_list.append(rolling_test_seq)

    rolling_test_set = ListDataset(
        rolling_test_seq_list, freq=freq, one_dim_target=False
    )
    return rolling_test_set


def df_to_mvds(df, freq='H'):
    datasets = []
    for variable in df.keys():
        ds = {"item_id" : variable, "target" : df[variable], "start": str(df.index[0])}
        datasets.append(ds)
    dataset = ListDataset(datasets,freq=freq)
    return dataset