import numpy as np
from .metrics import *

class Evaluator:
    
    def __init__(self, quantiles_num=10, smooth=False):
        self.quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
        self.ignore_invalid_values = True
        self.smooth = smooth

    def loss_name(self, q):
        return f"QuantileLoss[{q}]"

    def weighted_loss_name(self, q):
        return f"wQuantileLoss[{q}]"

    def coverage_name(self, q):
        return f"Coverage[{q}]"

    def get_metrics(self, targets, forecasts, samples_dim=0):
        mean_forecasts = forecasts.mean(axis=samples_dim)
        median_forecasts = np.quantile(forecasts, 0.5, axis=samples_dim)
        metrics = {
            "MSE": mse(targets, mean_forecasts),
            "abs_error": abs_error(targets, median_forecasts),
            "abs_target_sum": abs_target_sum(targets),
            "abs_target_mean": abs_target_mean(targets),
            "MAPE": mape(targets, median_forecasts),
            "sMAPE": smape(targets, median_forecasts),
        }
        
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]
        
        for q in self.quantiles:
            q_forecasts = np.quantile(forecasts, q, axis=samples_dim)
            metrics[self.loss_name(q)] = quantile_loss(targets, q_forecasts, q)
            metrics[self.weighted_loss_name(q)] = \
                metrics[self.loss_name(q)] / metrics["abs_target_sum"]
            metrics[self.coverage_name(q)] = coverage(targets, q_forecasts)
        
        metrics["mean_absolute_QuantileLoss"] = np.mean(
            [metrics[self.loss_name(q)] for q in self.quantiles]
        )
        metrics["CRPS"] = np.mean(
            [metrics[self.weighted_loss_name(q)] for q in self.quantiles]
        )
        metrics["MAE_Coverage"] = np.mean(
            [
                np.abs(metrics[self.coverage_name(q)] - np.array([q]))
                for q in self.quantiles
            ]
        )
        
        return metrics

    @property
    def selected_metrics(self):
        return ["CRPS", "ND", "NRMSE", "MSE"]

    def __call__(self, targets, forecasts):
        """

        Parameters
        ----------
        targets
            groundtruth in (batch_size, prediction_length, target_dim)
        forecasts
            forecasts in (batch_size, num_samples, prediction_length, target_dim)
        Returns
        -------
        Dict[String, float]
            metrics
        """
        targets = targets.cpu().detach().numpy()
        forecasts = forecasts.cpu().detach().numpy()
        if self.ignore_invalid_values:
            targets = np.ma.masked_invalid(targets)
            forecasts = np.ma.masked_invalid(forecasts)
        
        metrics = self.get_metrics(targets, forecasts, samples_dim=1)
        metrics_sum = self.get_metrics(targets.sum(axis=-1), forecasts.sum(axis=-1), samples_dim=1)
        
        # select output metrics
        output_metrics = dict()
        for k in self.selected_metrics:
            output_metrics[k] = metrics[k]
            output_metrics[f"{k}-Sum"] = metrics_sum[k]
        return output_metrics