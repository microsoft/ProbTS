# ---------------------------------------------------------------------------------
# Portions of this file are derived from Autoformer
# - Source: https://github.com/thuml/Autoformer
# - Paper: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
# - License: MIT License

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
from probts.model.forecaster import Forecaster
from probts.model.nn.layers.Embed import DataEmbedding_wo_pos
from probts.model.nn.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from probts.model.nn.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(Forecaster):
    def __init__(
        self,
        moving_avg: int = 25,
        factor: int = 1,
        n_heads: int = 8,
        activation: str = 'gelu',
        e_layers: int = 2,
        d_layers: int = 1,
        output_attention: bool = False,
        d_ff: int = 512,
        label_len: int = 48,
        embed: str = 'timeF',
        dropout: float = 0.1,
        f_hidden_size: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.label_len = self.context_length

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.target_dim, f_hidden_size, embed, self.freq.lower(),
                                                  dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.target_dim, f_hidden_size, embed, self.freq.lower(),
                                                  dropout)

        # Encoder
        self.model_encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        f_hidden_size, n_heads),
                    f_hidden_size,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(f_hidden_size)
        )
        
        # Decoder
        self.model_decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        f_hidden_size, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        f_hidden_size, n_heads),
                    f_hidden_size,
                    self.target_dim,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(f_hidden_size),
            projection=nn.Linear(f_hidden_size, self.target_dim, bias=True)
        )
        self.loss_fn = nn.MSELoss()
        
    def forward(self, inputs, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, *args, **kwargs):
        B, _, _ = inputs.shape

        if self.use_time_feat:
            past_target = inputs[:,:self.context_length, :self.target_dim]
            x_mark_enc = inputs[:,:self.context_length, self.target_dim:]
            time_feat = inputs[:,:,self.target_dim:]
        else:
            past_target = inputs[:,:self.context_length,:self.target_dim]
            x_mark_enc = None
            time_feat = None
            
        
        # decomp init
        mean = torch.mean(past_target, dim=1).unsqueeze(1).repeat(1, self.prediction_length, 1)
        zeros = torch.zeros([B, self.prediction_length, self.target_dim], device=past_target.device)
        seasonal_init, trend_init = self.decomp(past_target)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(past_target, x_mark_enc)
        enc_out, attns = self.model_encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, time_feat)
        seasonal_part, trend_part = self.model_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out[:, -self.prediction_length:, :]

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        outputs = self(inputs)
        
        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'all')

        outputs = self(inputs)
        return outputs.unsqueeze(1)