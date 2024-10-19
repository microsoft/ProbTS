import numpy as np
import torch
from torch import optim
from typing import Dict
import lightning.pytorch as pl
import sys
from probts.data import ProbTSBatchData
from probts.model.forecaster import Forecaster
from probts.utils import Evaluator, Scaler
from probts.utils.metrics import *
from probts.utils.save_utils import update_metrics, calculate_average, save_point_error, load_checkpoint, get_hor_str


def get_weights(sampling_weight_scheme, max_hor):
    '''
    return: w [max_hor]
    '''
    if sampling_weight_scheme == 'random':
        i_array = np.linspace(1 + 1e-5, max_hor - 1e-3, max_hor)
        w = (1 / max_hor) * (np.log(max_hor) - np.log(i_array))
    elif sampling_weight_scheme == 'const':
        w = np.array([1 / max_hor] * max_hor)
    elif sampling_weight_scheme == 'none':
        return None
    else:
        raise ValueError(f"Invalid sampling scheme {sampling_weight_scheme}.")
    
    return torch.tensor(w)
    

class HorizonReweightForecaster(pl.LightningModule):
    def __init__(
        self,
        forecaster: Forecaster,
        scaler: Scaler = None,
        pred_len_list: list = None,
        num_samples: int = 100,
        learning_rate: float = 1e-3,
        quantiles_num: int = 10,
        load_from_ckpt: str = None,
        save_point_error: bool = False,
        sampling_weight_scheme: str = 'none',
        **kwargs
    ):
        super().__init__()
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.load_from_ckpt = load_from_ckpt
        self.forecaster = forecaster
        
        self.max_hor = max(pred_len_list)
        self.min_hor = min(pred_len_list)
        self.save_point_error = save_point_error
            
        self.scaler = scaler
        self.evaluator = Evaluator(quantiles_num=quantiles_num)
        
        # init the parapemetr for sampling
        self.sampling_weight_scheme = sampling_weight_scheme

        print(f'sampling_weight_scheme: {sampling_weight_scheme}')

        self.save_hyperparameters()

    @classmethod
    def load_from_checkpoint(self, checkpoint_path, scaler=None, learning_rate=None, no_training=False, **kwargs):
        model = load_checkpoint(self, checkpoint_path, scaler=scaler, learning_rate=learning_rate, no_training=no_training, **kwargs)
        return model

    def training_forward(self, batch_data):
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        batch_data.future_target_cdf = self.scaler.transform(batch_data.future_target_cdf)
        
        loss = self.forecaster.loss(batch_data)

        loss_weights = get_weights(self.sampling_weight_scheme, self.max_hor)
        if loss_weights is not None:
            loss = (loss_weights.detach().to(loss.device).unsqueeze(0).unsqueeze(-1) * loss).sum(dim=1)

        loss = loss.mean()
        return loss
        

    def training_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        loss = self.training_forward(batch_data)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss


    def evaluate(self, batch, stage='',dataloader_idx=None):
        batch_data = ProbTSBatchData(batch, self.device)
        pred_len = batch_data.future_target_cdf.shape[1]
        orin_past_data = batch_data.past_target_cdf[:]
        orin_future_data = batch_data.future_target_cdf[:]

        norm_past_data = self.scaler.transform(batch_data.past_target_cdf)
        norm_future_data = self.scaler.transform(batch_data.future_target_cdf)
        
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)[:,:, :pred_len]
        
        # Calculate denorm metrics
        denorm_forecasts = self.scaler.inverse_transform(forecasts)
        
        # Calculate norm metrics
        norm_metrics = self.evaluator(norm_future_data, forecasts, past_data=norm_past_data, freq=self.forecaster.freq)
        self.metrics_dict = update_metrics(norm_metrics, stage, 'norm', target_dict=self.metrics_dict)
        
        
        l = orin_future_data.shape[1]
        
        if stage != 'test' and self.sampling_weight_scheme not in ['fix', 'none']:
            loss_weights = get_weights('random', l)
        else:
            loss_weights = None
            
        metrics = self.evaluator(orin_future_data, denorm_forecasts, past_data=orin_past_data, freq=self.forecaster.freq, loss_weights=loss_weights)
        self.metrics_dict = update_metrics(metrics, stage, target_dict=self.metrics_dict)
        
        if stage == 'test':
            hor_str = get_hor_str(self.forecaster.prediction_length, dataloader_idx)
            if hor_str not in self.hor_metrics:
                self.hor_metrics[hor_str] = {}

            if self.save_point_error:
                self.point_error = save_point_error(orin_future_data.cpu().detach().numpy(), 
                                                    denorm_forecasts[:,0,:].cpu().detach().numpy(), 
                                                    self.point_error, hor_str)
                
            self.hor_metrics[hor_str] = update_metrics(metrics, stage, target_dict=self.hor_metrics[hor_str])

        return metrics


    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        metrics = self.evaluate(batch, stage='val',dataloader_idx=dataloader_idx)
        
        return metrics


    def on_validation_epoch_start(self):
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.point_error = {}

    def on_validation_epoch_end(self):
        avg_metrics = calculate_average(self.metrics_dict)
        self.log_dict(avg_metrics, prog_bar=True) 


    def test_step(self, batch, batch_idx, dataloader_idx=None):
        metrics = self.evaluate(batch, stage='test',dataloader_idx=dataloader_idx)
        return metrics

    def on_test_epoch_start(self):
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.avg_metrics = {}
        self.avg_hor_metrics = {}
        self.point_error = {}

    def on_test_epoch_end(self):
        if len(self.hor_metrics) > 0:
            for hor_str, metric in self.hor_metrics.items():
                self.avg_hor_metrics[hor_str] = calculate_average(metric)
                self.avg_metrics.update(calculate_average(metric, hor=hor_str))
        else:
            self.avg_metrics = calculate_average(self.metrics_dict)
            

    def predict_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)
        return forecasts

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
