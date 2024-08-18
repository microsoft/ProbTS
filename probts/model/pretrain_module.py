from typing import Callable

from einops import rearrange, repeat

from probts.data import ProbTSBatchData
from probts.model.probts_module import ProbTSBaseModule


class ProbTSPretrainModule(ProbTSBaseModule):
    def __init__(
        self,
        dataloader_id_mapper: Callable = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mapper = dataloader_id_mapper

    def training_forward(self, batch_data):
        batch_ids = batch_data.dataset_idx
        batch_data.past_target_cdf = self.batch_scaler_transform(
            batch_data.past_target_cdf, batch_data.target_dimension_indicator, batch_ids
        )
        batch_data.future_target_cdf = self.batch_scaler_transform(
            batch_data.future_target_cdf, batch_data.target_dimension_indicator, batch_ids
        )

        loss = self.forecaster.loss(batch_data)
        return loss

    def training_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        loss = self.training_forward(batch_data)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def evaluate(self, batch, stage="", dataloader_idx=None):
        assert dataloader_idx is not None
        dataset_idx = self.mapper(dataloader_idx)

        batch_data = ProbTSBatchData(batch, self.device)
        batch_size = batch_data.past_target_cdf.shape[0]
        # self.batch_size.append(batch_size)

        orin_past_data = batch_data.past_target_cdf[:]
        orin_future_data = batch_data.future_target_cdf[:]
        
        scaler = self.scaler[dataset_idx]
        batch_data.past_target_cdf = scaler.transform(batch_data.past_target_cdf)
        batch_data.future_target_cdf = scaler.transform(batch_data.future_target_cdf)

        # pretrain: multivaraite -> univariate
        target_dim = batch_data.past_target_cdf.shape[-1]
        batch_data.past_target_cdf = rearrange(batch_data.past_target_cdf, 'b t c -> (b c) t 1')
        batch_data.past_is_pad = repeat(batch_data.past_is_pad, 'b t -> (b c) t', c=target_dim)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)
        forecasts = rearrange(forecasts, '(b c) s t 1 -> b s t c', b=batch_size)

        denorm_forecasts = scaler.inverse_transform(forecasts)

        metrics = self.evaluator(
            orin_future_data,
            denorm_forecasts,
            past_data=orin_past_data,
            freq=self.forecaster.freq,
        )
        self.metrics_dict = self.update_metrics(metrics, stage, target_dict=self.metrics_dict)

        # Calculate norm metrics
        norm_metrics = self.evaluator(
            batch_data.future_target_cdf,
            forecasts,
            past_data=batch_data.past_target_cdf,
            freq=self.forecaster.freq,
        )
        self.metrics_dict = self.update_metrics(norm_metrics, stage, 'norm', target_dict=self.metrics_dict)

        # if stage == 'test':
        #     if dataset_idx is not None:
        #         dataloader_str = str(self.forecaster.dataset[dataset_idx]) + f'_{dataloader_idx}'
        #     # elif type(self.forecaster.prediction_length) == list:  # noqa: E721
        #     #     dataloader_str = str(self.forecaster.prediction_length[0])
        #     else:
        #         dataloader_str = str(self.forecaster.dataset)
            
        #     if dataloader_str not in self.hor_metrics:
        #         self.hor_metrics[dataloader_str] = {}
            
        #     self.hor_metrics[dataloader_str] = self.update_metrics(metrics, stage, target_dict=self.hor_metrics[dataloader_str])
        #     self.hor_metrics[dataloader_str] = (self.update_metrics(norm_metrics, stage, 'norm', target_dict=self.hor_metrics[dataloader_str]))
            
        return metrics

    def batch_scaler_transform(
        self,
        batch_data,
        batch_dimension_indicator,
        batch_ids,
        inverse=False,
    ):
        if isinstance(self.scaler, list):
            for i, scaler in enumerate(self.scaler):
                mask = batch_ids == i
                if inverse:
                    batch_data[mask] = scaler.inverse_transform(batch_data[mask], batch_dimension_indicator[mask])
                else:
                    batch_data[mask] = scaler.transform(batch_data[mask], batch_dimension_indicator[mask])
        else:
            if inverse:
                batch_data = self.scaler.inverse_transform(batch_data)
            else:
                batch_data = self.scaler.transform(batch_data)
        return batch_data
