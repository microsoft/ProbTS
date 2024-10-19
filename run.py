import os
import json
import torch
import logging
from probts.data import ProbTSDataModule, ElasTSTDataModule 
from probts.model.forecast_module import ProbTSForecastModule
from probts.model.reweight_forecast_module import HorizonReweightForecaster
from probts.callbacks import MemoryCallback, TimeCallback
from probts.utils import find_best_epoch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from probts.utils.save_utils import save_exp_summary, save_csv

import warnings
warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('high')

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProbTSCli(LightningCLI):
    def add_arguments_to_parser(self, parser):
        data_to_model_link_args = [
            "scaler",
            "pred_len_list", 
        ]
        data_to_forecaster_link_args = [
            "target_dim",
            "history_length",
            "context_length",
            "prediction_length",
            "pred_len_list", 
            "lags_list",
            "freq",
            "time_feat_dim",
        ]
        for arg in data_to_model_link_args:
            parser.link_arguments(f"data.data_manager.{arg}", f"model.{arg}", apply_on="instantiate")
        for arg in data_to_forecaster_link_args:
            parser.link_arguments(f"data.data_manager.{arg}", f"model.forecaster.init_args.{arg}", apply_on="instantiate")

    def init_exp(self):
        config_args = self.parser.parse_args()
        self.tag = "_".join([
            self.datamodule.data_manager.dataset,
            self.model.forecaster.name,
            str(config_args.seed_everything),
            'CTX','-'.join([str(i) for i in self.datamodule.data_manager.ctx_len_list]),
            'PRED','-'.join([str(i) for i in self.datamodule.data_manager.pred_len_list]),
            'VALCTX','-'.join([str(i) for i in self.datamodule.data_manager.context_length]),
            'VALPRED','-'.join([str(i) for i in self.datamodule.data_manager.val_pred_len_list]),
        ])
        
        log.info(f"Root dir is {self.trainer.default_root_dir}, exp tag is {self.tag}")

        if not os.path.exists(self.trainer.default_root_dir):
            os.makedirs(self.trainer.default_root_dir)
            
        self.save_dict = f'{self.trainer.default_root_dir}/{self.tag}'
        if not os.path.exists(self.save_dict):
            os.makedirs(self.save_dict)

        if self.model.load_from_ckpt is not None:
            # if the checkpoint file is not assigned, find the best epoch in the current folder
            if '.ckpt' not in self.model.load_from_ckpt:
                _, best_ckpt = find_best_epoch(self.model.load_from_ckpt)
                print("find best ckpt ", best_ckpt)
                self.model.load_from_ckpt = os.path.join(self.model.load_from_ckpt, best_ckpt)
            
            # loading the checkpoints
            log.info(f"Loading pre-trained checkpoint from {self.model.load_from_ckpt}")
            self.model = HorizonReweightForecaster.load_from_checkpoint(
                self.model.load_from_ckpt,
                learning_rate=config_args.model.learning_rate,
                scaler=self.datamodule.data_manager.scaler,
                context_length=self.datamodule.data_manager.context_length,
                target_dim=self.datamodule.data_manager.target_dim,
                freq=self.datamodule.data_manager.freq,
                prediction_length=self.datamodule.data_manager.prediction_length,
                pred_len_list=self.datamodule.data_manager.pred_len_list,
                lags_list=self.datamodule.data_manager.lags_list,
                time_feat_dim=self.datamodule.data_manager.time_feat_dim,
                no_training=self.model.forecaster.no_training,
                save_point_error=self.model.save_point_error,
                sampling_weight_scheme=self.model.sampling_weight_scheme,
            )

        # Set callbacks
        callbacks = []
        
        if self.model.sampling_weight_scheme in ['none', 'fix']:
            monitor = 'val_CRPS'
        else:
            monitor = 'val_weighted_ND'
        
        if not self.model.forecaster.no_training:
            # Set callbacks
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=f'{self.save_dict}/ckpt',
                filename='{epoch}-{val_CRPS:.6f}',
                every_n_epochs=1,
                monitor=monitor,
                save_top_k=-1,
                save_last=True,
                enable_version_counter=False
            )
            callbacks.append(self.checkpoint_callback)

        self.memory_callback = MemoryCallback()
        self.time_callback = TimeCallback()
        
        callbacks.append(self.memory_callback)
        callbacks.append(self.time_callback)
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
            save_dir=f'{self.save_dict}/logs',
            name=self.tag,
            version='fit'
        )
    
    def set_test_mode(self):
        self.trainer.logger = CSVLogger(
            save_dir=f'{self.save_dict}/logs',
            name=self.tag,
            version='test'
        )

        if not self.model.forecaster.no_training:
            self.ckpt = self.checkpoint_callback.best_model_path
            log.info(f"Loading best checkpoint from {self.ckpt}")
            
            self.model = HorizonReweightForecaster.load_from_checkpoint(
                self.ckpt, 
                scaler=self.datamodule.data_manager.scaler,
                context_length=self.datamodule.data_manager.context_length,
                target_dim=self.datamodule.data_manager.target_dim,
                freq=self.datamodule.data_manager.freq,
                prediction_length=self.datamodule.data_manager.prediction_length,
                pred_len_list=self.datamodule.data_manager.pred_len_list,
                lags_list=self.datamodule.data_manager.lags_list,
                time_feat_dim=self.datamodule.data_manager.time_feat_dim,
                sampling_weight_scheme=self.model.sampling_weight_scheme,
            )
    
    def run(self):
        self.init_exp()
        
        if not self.model.forecaster.no_training:
            self.set_fit_mode()
            self.trainer.fit(model=self.model, datamodule=self.datamodule)
            inference=False
        else:
            inference=True

        self.set_test_mode()
        self.trainer.test(model=self.model, datamodule=self.datamodule)
        
        # saving multi testing horizon results
        save_exp_summary(self, inference=inference)
        save_csv(self.save_dict, self.model, self.datamodule.data_manager.context_length[0])


if __name__ == '__main__':
    cli = ProbTSCli(
        datamodule_class=ElasTSTDataModule,
        model_class=HorizonReweightForecaster,
        save_config_kwargs={"overwrite": True},
        run=False
    )
    cli.run()