import json
import logging
import os
import warnings

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from probts.callbacks import MemoryCallback, TimeCallback
from probts.data import ProbTSDataModule
from probts.model.forecast_module import ProbTSForecastModule
from probts.model.pretrain_module import ProbTSPretrainModule
from probts.utils.constant import DATA_TO_FORECASTER_ARGS, DATA_TO_MODEL_ARGS

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("high")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProbTSCli(LightningCLI):
    def __init__(self, *args, **kwargs):
        self.data_to_forecaster_link_args = list(DATA_TO_FORECASTER_ARGS)
        self.data_to_model_link_args = list(DATA_TO_MODEL_ARGS)
        if isinstance(kwargs["model_class"], ProbTSPretrainModule):
            self.data_to_model_link_args.append("dataloader_id_mapper")
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        for arg in self.data_to_model_link_args:
            parser.link_arguments(
                f"data.data_manager.{arg}", f"model.{arg}", apply_on="instantiate"
            )
        for arg in self.data_to_forecaster_link_args:
            parser.link_arguments(
                f"data.data_manager.{arg}",
                f"model.forecaster.init_args.{arg}",
                apply_on="instantiate",
            )

    def init_exp(self):
        config_args = self.parser.parse_args()
        self.tag = "_".join(
            [
                str(self.datamodule.data_manager),
                self.model.forecaster.name,
                str(config_args.seed_everything),
            ]
        )
        log.info(f"Root dir is {self.trainer.default_root_dir}, exp tag is {self.tag}")

        if not os.path.exists(self.trainer.default_root_dir):
            os.makedirs(self.trainer.default_root_dir)

        if self.model.load_from_ckpt is not None:
            log.info(f"Loading pre-trained checkpoint from {self.model.load_from_ckpt}")
            self.model = self.model_class.load_from_checkpoint(
                self.model.load_from_ckpt,
                learning_rate=config_args.model.learning_rate,
                scaler=self.datamodule.data_manager.scaler,
                context_length=self.datamodule.data_manager.context_length,
                target_dim=self.datamodule.data_manager.target_dim,
                freq=self.datamodule.data_manager.freq,
                prediction_length=self.datamodule.data_manager.prediction_length,
                lags_list=self.datamodule.data_manager.lags_list,
                time_feat_dim=self.datamodule.data_manager.time_feat_dim,
                no_training=self.model.forecaster.no_training,
                dataset=self.datamodule.data_manager.dataset,
            )

        # Set callbacks
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.trainer.default_root_dir}/ckpt/{self.tag}",
            filename="{epoch}-{val_CRPS:.3f}",
            every_n_epochs=1,
            monitor="val_CRPS",
            save_top_k=-1,
            save_last=True,
            enable_version_counter=False,
        )
        self.memory_callback = MemoryCallback()
        self.time_callback = TimeCallback()

        callbacks = [self.checkpoint_callback, self.memory_callback, self.time_callback]
        self.set_callbacks(callbacks)

    def set_callbacks(self, callbacks):
        # Replace built-in callbacks with custom callbacks
        custom_callbacks_name = [c.__class__.__name__ for c in callbacks]
        for c in self.trainer.callbacks:
            if c.__class__.__name__ in custom_callbacks_name:
                self.trainer.callbacks.remove(c)
        for c in callbacks:
            self.trainer.callbacks.append(c)
        for c in self.trainer.callbacks:
            if c.__class__.__name__ == "ModelSummary":
                self.model_summary_callback = c

    def set_fit_mode(self):
        self.trainer.logger = TensorBoardLogger(
            save_dir=f"{self.trainer.default_root_dir}/logs",
            name=self.tag,
            version="fit",
        )

    def set_test_mode(self):
        self.trainer.logger = CSVLogger(
            save_dir=f"{self.trainer.default_root_dir}/logs",
            name=self.tag,
            version="test",
        )

        if not self.model.forecaster.no_training:
            self.ckpt = self.checkpoint_callback.best_model_path
            log.info(f"Loading best checkpoint from {self.ckpt}")
            self.model = self.model_class.load_from_checkpoint(
                self.ckpt,
                scaler=self.datamodule.data_manager.scaler,
                mapper=self.datamodule.data_manager.dataloader_id_mapper,
                context_length=self.datamodule.data_manager.context_length,
                target_dim=self.datamodule.data_manager.target_dim,
                freq=self.datamodule.data_manager.freq,
                prediction_length=self.datamodule.data_manager.prediction_length,
                lags_list=self.datamodule.data_manager.lags_list,
                time_feat_dim=self.datamodule.data_manager.time_feat_dim,
                dataset=self.datamodule.data_manager.dataset,
            )

    def save_exp_summary(self):
        exp_summary = {}

        model_summary = self.model_summary_callback._summary(self.trainer, self.model)
        exp_summary["total_parameters"] = model_summary.total_parameters
        exp_summary["trainable_parameters"] = model_summary.trainable_parameters
        exp_summary["model_size"] = model_summary.model_size

        memory_summary = self.memory_callback.memory_summary
        exp_summary["memory_summary"] = memory_summary

        time_summary = self.time_callback.time_summary
        exp_summary["time_summary"] = time_summary
        for batch_key, batch_time in time_summary.items():
            if len(batch_time) > 0:
                exp_summary[f"mean_{batch_key}"] = sum(batch_time) / len(batch_time)

        summary_save_path = (
            f"{self.trainer.default_root_dir}/logs/{self.tag}/summary.json"
        )
        with open(summary_save_path, "w") as f:
            json.dump(exp_summary, f, indent=4)
        log.info(f"Summary saved to {summary_save_path}")

    def run(self):
        self.init_exp()

        if not self.model.forecaster.no_training:
            self.set_fit_mode()
            self.trainer.fit(model=self.model, datamodule=self.datamodule)

        self.set_test_mode()
        self.trainer.test(model=self.model, datamodule=self.datamodule)

        if not self.model.forecaster.no_training:
            self.save_exp_summary()


if __name__ == "__main__":
    cli = ProbTSCli(
        datamodule_class=ProbTSDataModule,
        # model_class=ProbTSForecastModule,
        model_class=ProbTSPretrainModule,
        save_config_kwargs={"overwrite": True},
        run=False,
    )
    cli.run()
