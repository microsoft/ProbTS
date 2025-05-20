# ---------------------------------------------------------------------------------
# Portions of this file are derived from LTSF-Linear
# - Source: https://github.com/cure-lab/LTSF-Linear
# - Paper: Are Transformers Effective for Time Series Forecasting?
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.decomp import series_decomp

class DLinear(Forecaster):
    def __init__(
        self,
        kernel_size: int,
        individual: bool,
        **kwargs
    ):
        super().__init__(**kwargs)
        if self.input_size != self.target_dim:
            self.enc_linear = nn.Linear(
                in_features=self.input_size, out_features=self.target_dim
            )
        else:
            self.enc_linear = nn.Identity()


        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.target_dim):
                self.Linear_Seasonal.append(nn.Linear(self.context_length, self.prediction_length))
                self.Linear_Trend.append(nn.Linear(self.context_length, self.prediction_length))
        else:
            self.Linear_Seasonal = nn.Linear(self.context_length, self.prediction_length)
            self.Linear_Trend = nn.Linear(self.context_length, self.prediction_length)
        self.loss_fn = nn.MSELoss(reduction='none')

    def encoder(self, inputs):
        seasonal_init, trend_init = self.decompsition(inputs)

        # [B,C,L]
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.prediction_length],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.prediction_length],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.target_dim):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        outputs = seasonal_output + trend_output # [B,C,L]
        return outputs.permute(0,2,1)

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        outputs = self.encoder(inputs)
        
        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        outputs = self.encoder(inputs)
        return outputs.unsqueeze(1)
