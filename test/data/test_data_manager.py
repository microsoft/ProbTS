import pytest

from probts.data.data_manager import DataManager
from probts.utils.constant import PROBTS_DATA_KEYS


ETTH1 = {
    "n_time_features": 4,
    "n_variables": 7,
}


@pytest.mark.parametrize(
    "dataset_list",
    [
        ["etth1"],
    ],
)  # ["etth1", "exchange_rate_nips"]
@pytest.mark.parametrize("history_length", [96])
@pytest.mark.parametrize("context_length", [96])
@pytest.mark.parametrize("prediction_length", [96])
def test_data_manager(dataset_list, history_length, context_length, prediction_length):
    manager = DataManager(
        dataset_list,
        history_length=history_length,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    train_iter = iter(manager.train_iter_dataset)
    val_iter = iter(manager.val_iter_dataset)
    test_iter = iter(manager.test_iter_dataset)

    TEST_EPOCH = 1
    for epoch in range(TEST_EPOCH):
        train_sample = next(train_iter)
        val_sample = next(val_iter)
        test_sample = next(test_iter)
    samples = [train_sample, val_sample, test_sample]

    # assert all keys are contained in the samples
    for key in PROBTS_DATA_KEYS:
        for sample in samples:
            assert key in sample.keys()

    # assert the shape of the samples
    for sample in samples:
        assert sample["past_time_feat"].shape == (
            history_length,
            ETTH1["n_time_features"],
        )
        assert sample["past_target_cdf"].shape == (history_length, ETTH1["n_variables"])
        assert sample["future_target_cdf"].shape == (
            prediction_length,
            ETTH1["n_variables"],
        )
        assert sample["future_time_feat"].shape == (
            prediction_length,
            ETTH1["n_time_features"],
        )


@pytest.mark.parametrize(
    "dataset_list",
    [
        ["etth1"],
    ],
)  # ["etth1", "exchange_rate_nips"]
@pytest.mark.parametrize("history_length", [96])
@pytest.mark.parametrize("context_length", [96])
@pytest.mark.parametrize("prediction_length", [96])
def test_data_manager_univariate(
    dataset_list, history_length, context_length, prediction_length
):
    manager = DataManager(
        dataset_list,
        history_length=history_length,
        context_length=context_length,
        prediction_length=prediction_length,
        is_pretrain=True,  # key difference here!!!
    )
    train_iter = iter(manager.train_iter_dataset)
    val_iter = iter(manager.val_iter_dataset)
    test_iter = iter(manager.test_iter_dataset)

    TEST_EPOCH = 1
    for epoch in range(TEST_EPOCH):
        train_sample = next(train_iter)
        val_sample = next(val_iter)
        test_sample = next(test_iter)
    samples = [train_sample, val_sample, test_sample]

    # assert all keys are contained in the samples
    for key in PROBTS_DATA_KEYS:
        for sample in samples:
            assert key in sample.keys()

    # assert the shape of the samples
    for sample in [train_sample]:
        assert sample["past_time_feat"].shape == (
            history_length,
            ETTH1["n_time_features"],
        )
        assert sample["past_target_cdf"].shape == (history_length, 1)
        assert sample["future_target_cdf"].shape == (prediction_length, 1)
        assert sample["future_time_feat"].shape == (
            prediction_length,
            ETTH1["n_time_features"],
        )

    for sample in [val_sample, test_sample]:
        assert sample["past_time_feat"].shape == (
            history_length,
            ETTH1["n_time_features"],
        )
        assert sample["past_target_cdf"].shape == (history_length, ETTH1["n_variables"])
        assert sample["future_target_cdf"].shape == (
            prediction_length,
            ETTH1["n_variables"],
        )
        assert sample["future_time_feat"].shape == (
            prediction_length,
            ETTH1["n_time_features"],
        )
