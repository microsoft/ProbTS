# ---------------------------------------------------------------------------------
# Portions of this file are derived from Chronos
# - Source: https://github.com/amazon-science/chronos-forecasting
# - Paper: Chronos: Learning the Language of Time Series
# - License: Apache License 2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
# from chronos import ChronosPipeline
from einops import rearrange
from probts.model.nn.arch.ChronosModule.base import BaseChronosPipeline
from probts.model.forecaster import Forecaster


class Chronos(Forecaster):
    def __init__(
        self,
        model_size: str = 'base',
        **kwargs
    ):
        super().__init__(**kwargs)

        if type(self.prediction_length) == list:
            self.prediction_length = max(self.prediction_length)
            

        if type(self.context_length) == list:
            self.context_length = max(self.context_length)
            
        self.pred_len = self.prediction_length

        # Load pretrained model
        self.no_training = True

        self.pipeline = BaseChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{model_size}",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
            device_map="cuda", 
            torch_dtype=torch.bfloat16,)
        
        self.q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Quantile levels



    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = inputs[:, -self.context_length:]
        
        B, _, K = inputs.shape
        inputs = rearrange(inputs, 'b l k -> (b k) l')#.cpu()
        context = [inputs[i] for i in range(B*K)]
        inner_batch_size = 12 # for 80G gpu
        forecast_samples = []

        # Process in batches of size `inner_batch_size`
        for i in range(0, len(context), inner_batch_size):
            batch_context = context[i:i + inner_batch_size]
            batch_forecast_samples = self.pipeline.predict(
                batch_context,
                prediction_length=self.pred_len,
                num_samples=num_samples,
                limit_prediction_length=False
            )
            forecast_samples.append(batch_forecast_samples)
        
        forecast_samples = torch.cat(forecast_samples, dim=0)
        prob_forecast = rearrange(forecast_samples, '(b k) s l -> b s l k', b=B, k=K)
        
        return prob_forecast


