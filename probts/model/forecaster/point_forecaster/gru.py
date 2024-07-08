import torch
import torch.nn as nn

from probts.data import ProbTSBatchData
from probts.utils import repeat
from probts.model.forecaster import Forecaster


class GRUForecaster(Forecaster):
    def __init__(
        self,
        num_layers: int = 2,
        f_hidden_size: int = 40,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.autoregressive = True
        
        self.model = nn.GRU(
            input_size=self.input_size,
            hidden_size=f_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(f_hidden_size, self.target_dim)
        self.loss_fn = nn.MSELoss(reduction='none')

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        outputs, _ = self.model(inputs)
        outputs = outputs[:, -self.prediction_length-1:-1, ...]
        outputs = self.linear(outputs)
        
        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        forecasts = []
        states = self.encode(batch_data)
        past_target_cdf = batch_data.past_target_cdf
        
        for k in range(self.prediction_length):
            current_batch_data = ProbTSBatchData({
                'target_dimension_indicator': batch_data.target_dimension_indicator,
                'past_target_cdf': past_target_cdf,
                'future_time_feat': batch_data.future_time_feat[:, k : k + 1:, ...]
            }, device=batch_data.device)

            outputs, states = self.decode(current_batch_data, states)
            outputs = self.linear(outputs)
            forecasts.append(outputs)

            past_target_cdf = torch.cat(
                (past_target_cdf, outputs), dim=1
            )

        forecasts = torch.cat(forecasts, dim=1).reshape(
            -1, self.prediction_length, self.target_dim)
        return forecasts.unsqueeze(1)

    def encode(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        outputs, states = self.model(inputs)
        return states

    def decode(self, batch_data, states=None):
        inputs = self.get_inputs(batch_data, 'decode')
        outputs, states = self.model(inputs, states)
        return outputs, states
