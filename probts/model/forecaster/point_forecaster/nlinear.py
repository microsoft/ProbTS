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


class NLinear(Forecaster):
    def __init__(
        self,
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

        self.target_dim = self.target_dim
        self.individual = individual
        if individual:
            self.Linear = nn.ModuleList()
            for i in range(self.target_dim):
                self.Linear.append(nn.Linear(self.context_length,self.prediction_length))
        else:
            self.Linear = nn.Linear(self.context_length, self.prediction_length)
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, inputs):
        seq_last = inputs[:,-1:,:].detach()
        inputs = inputs - seq_last
        if self.individual:
            output = torch.zeros([inputs.size(0),self.prediction_length,inputs.size(2)],dtype=inputs.dtype).to(inputs.device)
            for i in range(self.target_dim):
                output[:,:,i] = self.Linear[i](inputs[:,:,i])
        else:
            output = self.Linear(inputs.permute(0,2,1)).permute(0,2,1)
        output = output + seq_last
        return output

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        inputs = inputs[:, : self.context_length, ...]
        inputs = self.enc_linear(inputs)
        outputs = self(inputs)
        
        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        outputs = self(inputs)
        return outputs.unsqueeze(1)
