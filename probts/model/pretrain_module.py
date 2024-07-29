from probts.data import ProbTSBatchData
from probts.model.probts_module import ProbTSBaseModule
from einops import rearrange


class ProbTSPretrainModule(ProbTSBaseModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def training_forward(self, batch_data):
        batch_ids = batch_data.dataset_idx
        batch_data.past_target_cdf = self.batch_scaler_transform(
            batch_data.past_target_cdf, batch_ids
        )
        batch_data.future_target_cdf = self.batch_scaler_transform(
            batch_data.future_target_cdf, batch_ids
        )

        loss = self.forecaster.loss(batch_data)
        return loss

    def training_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        loss = self.training_forward(batch_data)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def evaluate(self, batch, stage="", dataloader_idx=None):
        batch_data = ProbTSBatchData(batch, self.device)
        batch_size = batch_data.past_target_cdf.shape[0]
        self.batch_size.append(batch_size)

        orin_past_data = batch_data.past_target_cdf[:]
        orin_future_data = batch_data.future_target_cdf[:]
  
        assert dataloader_idx is not None
        scaler = self.scaler[dataloader_idx]
        batch_data.past_target_cdf = scaler.transform(batch_data.past_target_cdf)
        batch_data.future_target_cdf = scaler.transform(batch_data.future_target_cdf)

        # pretrain: multivaraite -> univariate
        batch_data.past_target_cdf = rearrange(batch_data.past_target_cdf, 'b t c -> (b c) t 1')
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)
        forecasts = rearrange(forecasts, '(b c) s t 1 -> b s t c', b=batch_size)

        denorm_forecasts = scaler.transform(forecasts)

        metrics = self.evaluator(
            orin_future_data,
            denorm_forecasts,
            past_data=orin_past_data,
            freq=self.forecaster.freq,
        )
        self.update_metrics(metrics, stage)

        # Calculate norm metrics
        norm_metrics = self.evaluator(
            batch_data.future_target_cdf,
            forecasts,
            past_data=batch_data.past_target_cdf,
            freq=self.forecaster.freq,
        )
        self.update_metrics(norm_metrics, stage, "norm")
        return metrics


    def batch_scaler_transform(
        self,
        batch_data,
        batch_ids,
        inverse=False,
    ):
        if isinstance(self.scaler, list):
            for i, scaler in enumerate(self.scaler):
                mask = batch_ids == i
                if inverse:
                    batch_data[mask] = scaler.inverse_transform(batch_data[mask])
                else:
                    batch_data[mask] = scaler.transform(batch_data[mask])
        else:
            if inverse:
                batch_data = self.scaler.inverse_transform(batch_data)
            else:
                batch_data = self.scaler.transform(batch_data)
        return batch_data
