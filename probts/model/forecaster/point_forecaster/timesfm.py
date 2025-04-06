# ---------------------------------------------------------------------------------
# Portions of this file are derived from timesfm
# - Source: https://github.com/google-research/timesfm
# - Paper: A decoder-only foundation model for time-series forecasting
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import numpy as np
import torch
from einops import rearrange
import sys
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.TimesFMModule import TimesFm, TimesFmCheckpoint, TimesFmHparams
# from submodules.timesfm.src.timesfm import TimesFm

class TimesFM(Forecaster):
    def __init__(
        self,
        model_size: str = '200m',
        # input_patch_len: int = 32,
        # output_patch_len: int = 128,
        # num_layers: int = 20,
        # model_dims: int = 1280,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.no_training = True
        
        if (type(self.target_dim).__name__=='dict'):
            for dataset_name in self.target_dim:
                target_dim = target_dim[dataset_name]
                freq = freq[dataset_name]
        else:
            freq = self.freq
                
        if (type(self.context_length).__name__=='list'):
            context_length = max(context_length)
            
        if (type(self.prediction_length).__name__=='list'):
            prediction_length = max(prediction_length)
            
        if model_size not in ['200m', '500m']:
            print('Invalid model size. Please choose from 200m or 500m')
            sys.exit()

        if model_size == '200m':
            self.tfm = TimesFm(
                hparams=TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=128,
                ),
                checkpoint=TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
            )
        elif model_size == '500m':
            self.tfm = TimesFm(
                hparams=TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=32,
                    horizon_len=128,
                    num_layers=50,
                    use_positional_embedding=False,
                    context_len=2048,
                ),
                checkpoint=TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
            )

        
        freq_dict = {'h': 0, 'min': 0, 'd': 0, 'b': 0, 'u': 0, 'w': 1, 'm': 1, 'q': 2, 'y': 2}
        freq = freq.lower()
        
        if freq in freq_dict:
            self.freq_int = freq_dict[freq]
        else:
            self.freq_int = 0

        print(f"TimesFM-{model_size} - frequency: {freq}, freq_num: {self.freq_int}")


    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = inputs[:, -self.context_length:].cpu()
        B, _, K = inputs.shape
        # past_target = batch_data.past_target_cdf[:, -self.context_length:]
        
        inputs = np.array(rearrange(inputs, 'b l k -> (b k) l'))
        frequency_input = [self.freq_int] * inputs.shape[0]
        
        _, out = self.tfm.forecast(
            inputs,
            freq=frequency_input,
        )
        point_forecast = out[:, :, 5]
        point_forecast = rearrange(point_forecast, '(b k) l -> b l k', b=B,k=K)
        
        point_forecast = torch.tensor(point_forecast[:, :self.prediction_length])
        return point_forecast.unsqueeze(1)
    