import lightning.pytorch as pl
from torch.utils.data import DataLoader
from .data_manager import DataManager


class ProbTSDataModule(pl.LightningDataModule):
    r"""
    DataModule for probablistic time series datasets.
    """

    def __init__(
        self,
        data_manager: DataManager,
        batch_size: int = 64,
        test_batch_size: int = 8,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage: str):
        self.dataset_train = self.data_manager.train_iter_dataset
        self.dataset_val = self.data_manager.val_iter_dataset
        self.dataset_test = self.data_manager.test_iter_dataset

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.dataset_val, batch_size=self.test_batch_size, num_workers=1
        )
        return val_dataloader

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.test_batch_size, num_workers=1
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.test_batch_size, num_workers=1
        )
