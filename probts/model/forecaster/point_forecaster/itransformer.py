# ---------------------------------------------------------------------------------
# Portions of this file are derived from iTransformer
# - Source: https://github.com/thuml/iTransformer
# - Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
# - License: MIT License

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
from probts.model.forecaster import Forecaster
from probts.model.nn.layers.Transformer_EncDec import Encoder, EncoderLayer
from probts.model.nn.layers.SelfAttention_Family import FullAttention, AttentionLayer
from probts.model.nn.layers.Embed import DataEmbedding_inverted

class iTransformer(Forecaster):
    def __init__(
        self,
        factor: int = 1,
        n_heads: int = 8,
        activation: str = 'gelu',
        e_layers: int = 2,
        output_attention: bool = False,
        d_ff: int = 512,
        label_len: int = 48,
        use_norm: bool = True,
        class_strategy:str = 'projection',
        dropout: float = 0.1,
        f_hidden_size: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.label_len = label_len
        
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.context_length, f_hidden_size,
                                                    dropout)
        self.class_strategy = class_strategy
        # Encoder-only architecture
        self.model_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), f_hidden_size, n_heads),
                    f_hidden_size,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(f_hidden_size)
        )
        self.projector = nn.Linear(f_hidden_size, self.prediction_length, bias=True)
        self.loss_fn = nn.MSELoss()

    def forward(self, inputs):
        if self.use_time_feat:
            past_target = inputs[:,:,:self.target_dim]
            x_mark_enc = inputs[:,:,-self.target_dim:]
        else:
            past_target = inputs
            x_mark_enc = None
            
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = past_target.mean(1, keepdim=True).detach()
            past_target = past_target - means
            stdev = torch.sqrt(torch.var(past_target, dim=1, keepdim=True, unbiased=False) + 1e-5)
            past_target /= stdev

        _, _, N = past_target.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(past_target, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.model_encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.prediction_length, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.prediction_length, 1))

        return dec_out[:, -self.prediction_length:, :]

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        output = self(inputs)

        return output.unsqueeze(1)

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        outputs = self(inputs)
        
        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()