import torch
from einops import repeat
from probts.model.forecaster import Forecaster
import sys

class NaiveForecaster(Forecaster):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.no_training = True


    def forecast(self, batch_data, num_samples=None):
        last_value = batch_data.past_target_cdf[:,-1,:]
        outputs = repeat(last_value,'b k -> b n l k', n=1, l=self.prediction_length)
        return outputs
