import importlib
from typing import Dict, Union, List

import lightning.pytorch as pl
import numpy as np
import torch
from torch import optim

from probts.data import ProbTSBatchData
from probts.model.forecaster import Forecaster
from probts.utils import Evaluator, Scaler


class ProbTSBaseModule(pl.LightningModule):
    def __init__(
        self,
        forecaster: Forecaster,
        scaler: Union[Scaler, List[Scaler]] = None,
        num_samples: int = 100,
        learning_rate: float = 1e-3,
        quantiles_num: int = 10,
        load_from_ckpt: str = None,
        **kwargs,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.load_from_ckpt = load_from_ckpt

        self.forecaster = forecaster

        self.scaler = scaler
        self.evaluator = Evaluator(quantiles_num=quantiles_num)
        self.save_hyperparameters()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        scaler=None,
        learning_rate=None,
        no_training=False,
        **kwargs,
    ):
        # Load the checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
        # Extract the arguments for the forecaster
        forecaster_args = checkpoint["hyper_parameters"]["forecaster"]

        if isinstance(forecaster_args, Forecaster):
            forecaster = forecaster_args
        else:
            module_path, class_name = forecaster_args["class_path"].rsplit(".", 1)
            forecaster_class = getattr(importlib.import_module(module_path), class_name)

            # Add any missing required arguments
            forecaster_args = forecaster_args["init_args"]
            forecaster_args.update(kwargs)

            # Create the forecaster
            forecaster = forecaster_class(**forecaster_args)

        if learning_rate is None:
            learning_rate = checkpoint["hyper_parameters"].get("learning_rate", 1e-3)

        forecaster.no_training = no_training

        # Create the model instance
        model = cls(
            forecaster=forecaster,
            scaler=scaler,
            num_samples=checkpoint["hyper_parameters"].get("num_samples", 100),
            learning_rate=learning_rate,
            quantiles_num=checkpoint["hyper_parameters"].get("quantiles_num", 10),
            load_from_ckpt=checkpoint["hyper_parameters"].get("load_from_ckpt", None),
            **kwargs,  # Pass additional arguments here
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def training_forward(self, batch_data):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def evaluate(self, batch, stage=""):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        metrics = self.evaluate(batch, stage="val", dataloader_idx=dataloader_idx)
        return metrics

    def on_validation_epoch_start(self):
        self.metrics_dict = {}
        self.hor_metrics = {}

    def on_validation_epoch_end(self):
        avg_metrics = self.calculate_average(self.metrics_dict)
        self.log_dict(avg_metrics, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        metrics = self.evaluate(batch, stage="test", dataloader_idx=dataloader_idx)
        return metrics

    def on_test_epoch_start(self):
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.avg_metrics = {}
        self.avg_hor_metrics = {}

    def on_test_epoch_end(self):
        # avg_metrics = {}
        if len(self.hor_metrics) > 0:
            # for idx, hor in enumerate(self.forecaster.prediction_length):
            for hor_str, metric in self.hor_metrics.items():
                # self.avg_hor_metrics[str(hor)] = self.calculate_weighted_average(self.hor_metrics[str(hor)], self.batch_size[str(hor)])
                self.avg_hor_metrics[hor_str] = self.calculate_average(metric)
                self.avg_metrics.update(self.calculate_average(metric, hor=hor_str))
        else:
            self.avg_metrics = self.calculate_average(self.metrics_dict)
            # self.log_dict(self.avg_metrics, prog_bar=True)
        # TODO: fix bug, log_dict not seperate dataloaders into multiple columns
        self.log_dict(self.avg_metrics, logger=True)

    def predict_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)
        return forecasts

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        return optimizer

    def update_metrics(
        self, new_metrics: Dict, stage: str, key: str = "", target_dict={}
    ):
        prefix = stage if key == "" else f"{stage}_{key}"
        for metric_name, metric_value in new_metrics.items():
            metric_key = f"{prefix}_{metric_name}"
            if metric_key not in target_dict:
                target_dict[metric_key] = []

            if isinstance(metric_value, list):
                target_dict[metric_key] = target_dict[metric_key] + metric_value
            else:
                target_dict[metric_key].append(metric_value)

        return target_dict

    def calculate_weighted_average(self, metrics_dict: Dict, batch_size: list):
        metrics = {}
        for key, value in metrics_dict.items():
            metrics[key] = np.sum(value * np.array(batch_size)) / np.sum(batch_size)
        return metrics

    def calculate_average(self, metrics_dict: Dict, hor=""):
        metrics = {}
        if hor != "":
            hor = hor + "/"
        # print("metrics_dict: ", metrics_dict)
        for key, value in metrics_dict.items():
            metrics[hor + key] = np.mean(value)
        return metrics
