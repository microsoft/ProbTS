# ---------------------------------------------------------------------------------
# Portions of this file are derived from uni2ts
# - Source: https://github.com/SalesforceAIResearch/uni2ts
# - Paper: Unified Training of Universal Time Series Forecasting Transformers
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

from dataclasses import dataclass, field
from pathlib import Path

from gluonts.dataset.common import ListDataset
from datasets import load_from_disk
from datasets.features import Sequence
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from .probts_datasets import ProbTSDataset


@dataclass
class LotsaTSDatasetLoader(ProbTSDataset):
    # currently train and test are the same dataset
    freq: str = field(default=None)

    def __post_init__(self):
        super().__post_init__()

        self.__read_data()

        # self.target_dim = len(self.dataset_raw)
        freq_from_dataset = self.dataset_raw[0]["freq"]
        if self.freq is None or self.freq != freq_from_dataset:
            print(
                f"Warning: freq is not set or does not match the dataset {self.dataset}. Setting freq to {freq_from_dataset}"
            )
            self.freq = freq_from_dataset
        gluonts_list_dataset = self.hf_dataset_to_mvds(self.dataset_raw)
        grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        self.multi_group_set = grouper(gluonts_list_dataset) # returns a list of dict
        # print(self.multi_group_set[0]['target'].shape)

    def __read_data(self):
        self.dataset_raw = load_from_disk(Path(self.path) / self.dataset)
        self.features = dict(self.dataset_raw.features)
        self.non_seq_cols = [
            name
            for name, feat in self.features.items()
            if not isinstance(feat, Sequence)
        ]
        self.seq_cols = [
            name for name, feat in self.features.items() if isinstance(feat, Sequence)
        ]
        # TODO: tmp fix, remove other cols like 'past_feat_dynamic_real'
        assert "target" in self.seq_cols
        self.seq_cols = ["target"]
        # self.dataset_raw.set_format("numpy", columns=self.non_seq_cols)

    def hf_dataset_to_mvds(self, dataset, key_field="target"):
        datasets = []
        for i in range(len(dataset)):
            data = dataset[i]
            # key_field might contain more than 1 series, seperate them
            for j in range(
                len(data[key_field]) if isinstance(data[key_field][0], list) else 1
            ):
                data_item = {
                    "item_id": data["item_id"],
                    "start": data["start"],
                }
                data_item.update(
                    {
                        name: data[name][j]
                        if isinstance(data[key_field][0], list)
                        else data[name]
                        for name in self.seq_cols
                    }
                )
                datasets.append(data_item)
        self.target_dim = len(datasets)
        print(f"update target_dim: {self.target_dim}")
        return ListDataset(datasets, freq=self.freq)

    def get_iter_dataset(self, mode: str):
        if mode == "train":
            return super().get_iter_dataset(self.multi_group_set, mode)
        elif mode == "val":
            return super().get_iter_dataset(self.multi_group_set, mode)
        elif mode == "test":
            return super().get_iter_dataset(self.multi_group_set, mode)


@dataclass
class LotsaUniTSDatasetLoader(LotsaTSDatasetLoader):
    dataset_raw: ListDataset = field(default=None)
    freq: str = field(default=None)

    def __post_init__(self):
        ProbTSDataset.__post_init__(self)
        
        assert self.dataset_raw is not None, "dataset_raw need to be provided!"
        self.target_dim = len(self.dataset_raw)
        freq_from_dataset = self.dataset_raw[0]["freq"]
        if self.freq is None or self.freq != freq_from_dataset:
            # print(
            #     f"Warning: freq is not set or does not match the dataset {self.dataset}. Setting freq to {freq_from_dataset}"
            # )
            self.freq = freq_from_dataset
        gluonts_list_dataset = self.dataset_raw
        grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        self.multi_group_set = grouper(gluonts_list_dataset) # returns a list of dict
        # print(self.multi_group_set[0]['target'].shape)