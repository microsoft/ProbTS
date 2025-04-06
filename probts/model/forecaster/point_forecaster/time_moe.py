# ---------------------------------------------------------------------------------
# Portions of this file are derived from Time-MoE
# - Source: https://github.com/Time-MoE/Time-MoE
# - Paper: Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModelForCausalLM
from probts.model.forecaster import Forecaster
import sys
from probts.data.data_utils.data_scaler import InstanceNorm

class TimeMoE(Forecaster):
    def __init__(
        self,
        model_size: str = '50M',
        instance_norm=True,
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
            
        if model_size not in ['50M', '200M']:
            print('Invalid model size. Please choose from 50M or 200M')
            sys.exit()
        
        if instance_norm:
            self.normalization = InstanceNorm()
        else:
            self.normalization = None
            
        self.model = AutoModelForCausalLM.from_pretrained(
            f'Maple728/TimeMoE-{model_size}',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print(f"loaded TimeMoE-{model_size} model")
        

    def forecast(self, batch_data, num_samples=None):
        inputs = batch_data.past_target_cdf[:, -self.context_length:]
        # inputs = inputs[:, -self.context_length:].cpu()
        B, _, K = inputs.shape
        inputs = inputs.to(dtype=torch.bfloat16)
        inputs = rearrange(inputs, 'b l k -> (b k) l')
        
        if self.normalization:
            inputs = self.normalization(inputs, mode='norm')
            
        forecasts = self.model.generate(inputs, max_new_tokens=self.prediction_length)  # shape is [batch_size, 12 + 6]
        point_forecast = forecasts[:, -self.prediction_length:]
        
        
        if self.normalization:
            point_forecast = self.normalization(point_forecast, mode='denorm')
            
        point_forecast = point_forecast.to(dtype=torch.float32)
        point_forecast = rearrange(point_forecast, '(b k) l -> b l k', b=B,k=K)
        
        point_forecast = point_forecast[:, :self.prediction_length]
        return point_forecast.unsqueeze(1)
    