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


class LinearForecaster(Forecaster):
    def __init__(
        self,
        individual: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.individual = individual
        
        if self.individual:
            self.linear = nn.ModuleList()
            for i in range(self.input_size):
                self.linear.append(nn.Linear(self.context_length, self.prediction_length))
        else:
            self.linear = nn.Linear(self.context_length, self.prediction_length)
        self.out_linear = nn.Linear(self.input_size, self.target_dim)
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x):
        if self.individual:
            outputs = torch.zeros([x.size(0), self.prediction_length, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.input_size):
                outputs[:, :, i] = self.linear[i](x[:, :, i])
        else:
            outputs = self.linear(x.permute(0,2,1)).permute(0,2,1)
        outputs = self.out_linear(outputs)
        return outputs

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        forecasts = self(inputs).unsqueeze(1)
        return forecasts

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        outputs = self(inputs)
        
        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()
