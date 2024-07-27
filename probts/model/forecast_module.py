from probts.data import ProbTSBatchData
from probts.model.probts_module import ProbTSBaseModule


class ProbTSForecastModule(ProbTSBaseModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def training_forward(self, batch_data):
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        batch_data.future_target_cdf = self.scaler.transform(
            batch_data.future_target_cdf
        )
        loss = self.forecaster.loss(batch_data)
        return loss

    def training_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        loss = self.training_forward(batch_data)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def evaluate(self, batch, stage=""):
        batch_data = ProbTSBatchData(batch, self.device)
        orin_past_data = batch_data.past_target_cdf[:]
        orin_future_data = batch_data.future_target_cdf[:]
        norm_past_data = self.scaler.transform(batch_data.past_target_cdf)
        norm_future_data = self.scaler.transform(batch_data.future_target_cdf)
        self.batch_size.append(orin_past_data.shape[0])

        # Forecast
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)

        # Calculate denorm metrics
        denorm_forecasts = self.scaler.inverse_transform(forecasts)
        metrics = self.evaluator(
            orin_future_data,
            denorm_forecasts,
            past_data=orin_past_data,
            freq=self.forecaster.freq,
        )
        self.update_metrics(metrics, stage)

        # Calculate norm metrics
        norm_metrics = self.evaluator(
            norm_future_data,
            forecasts,
            past_data=norm_past_data,
            freq=self.forecaster.freq,
        )
        self.update_metrics(norm_metrics, stage, "norm")
        return metrics
    