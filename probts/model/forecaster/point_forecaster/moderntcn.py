# ---------------------------------------------------------------------------------
# Portions of this file are derived from ModernTCN
# - Source: https://github.com/luodhhh/ModernTCN/tree/main
# - Paper: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
# - License: MIT License
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------
import sys
import torch
import torch.nn as nn
from typing import List
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.decomp import series_decomp
from probts.model.nn.arch.ModernTCN_backbone import ModernTCNModel
# torch.backends.cudnn.enabled = False

class ModernTCN(Forecaster):
    def __init__(
        self,
        kernel_size: int = 25,             
        decomposition: int = 0,           
        stem_ratio: int = 6,             
        downsample_ratio: int = 2,      
        ffn_ratio: int = 2,          
        num_blocks: List[int] = [1, 1, 1, 1],  
        large_size: List[int] = [31, 29, 27, 13], 
        small_size: List[int] = [5, 5, 5, 5],  
        dims: List[int] = [256, 256, 256, 256], 
        dw_dims: List[int] = [256, 256, 256, 256], 
        small_kernel_merged: bool = False, 
        use_multi_scale: bool = True,     
        revin: int = 1,                  
        affine: int = 0,     
        subtract_last: int = 0,  
        individual: int = 0,      
        patch_size: int = 16,  
        patch_stride: int = 8,  
        dropout: float = 0.05,
        head_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.stem_ratio = stem_ratio
        self.downsample_ratio = downsample_ratio
        self.ffn_ratio = ffn_ratio
        self.num_blocks = num_blocks
        self.large_size = large_size
        self.small_size = small_size
        self.dims = dims
        self.dw_dims = dw_dims

        self.nvars = self.target_dim
        self.small_kernel_merged = small_kernel_merged
        self.drop_backbone = dropout
        self.drop_head = head_dropout
        self.use_multi_scale = use_multi_scale
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last

        self.seq_len = self.context_length
        self.c_in = self.nvars,
        self.individual = individual
        self.target_window = self.prediction_length

        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(self.kernel_size)
            self.model_res = ModernTCNModel(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)
            self.model_trend = ModernTCNModel(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)
        else:
            self.model = ModernTCNModel(patch_size=self.patch_size,patch_stride=self.patch_stride,stem_ratio=self.stem_ratio, downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_blocks=self.num_blocks, large_size=self.large_size, small_size=self.small_size, dims=self.dims, dw_dims=self.dw_dims,
                 nvars=self.nvars, small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head, use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                 subtract_last=self.subtract_last, freq=self.freq, seq_len=self.seq_len, c_in=self.c_in, individual=self.individual, target_window=self.target_window)
            
        self.loss_fn = nn.MSELoss(reduction='none')
        
        if self.input_size != self.target_dim:
            self.enc_linear = nn.Linear(
                in_features=self.input_size, out_features=self.target_dim
            )
        else:
            self.enc_linear = nn.Identity()

    def encoder(self, x, te=None):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            res = self.model_res(res_init, te)
            trend = self.model_trend(trend_init, te)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)

            x = self.model(x, te)
            x = x.permute(0, 2, 1)
        return x

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        # inputs = inputs[:,:,:self.target_dim]
        inputs = self.enc_linear(inputs)
        outputs = self.encoder(inputs)
        
        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        # b l k
        inputs = self.get_inputs(batch_data, 'encode')
        # inputs = inputs[:,:,:self.target_dim]
        inputs = self.enc_linear(inputs)
        outputs = self.encoder(inputs)
        return outputs.unsqueeze(1)
