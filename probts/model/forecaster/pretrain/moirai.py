# ---------------------------------------------------------------------------------
# Portions of this file are derived from uni2ts
# - Source: https://github.com/SalesforceAIResearch/uni2ts
# - Paper: Unified Training of Universal Time Series Forecasting Transformers
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from typing import Union

import torch
from einops import rearrange, repeat
from torch.distributions import Distribution

from uni2ts.distribution import (
    LogNormalOutput,
    MixtureOutput,
    NegativeBinomialOutput,
    NormalFixedScaleOutput,
    StudentTOutput,
)
from uni2ts.model.moirai.module import MoiraiModule

from probts.model.forecaster import Forecaster
from probts.model.nn.layers.Moirai_backbone import MoiraiBackbone
from probts.utils import weighted_average


class Moirai(Forecaster):
    def __init__(
        self,
        variate_mode: str = "M",
        patch_size: Union[str, int] = "auto",
        model_size: str = "base",
        scaling: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self.variate_mode = variate_mode
        # patch_size = patch_size if patch_size == 'auto' else int(patch_size)

        mixture_output = MixtureOutput(
            components=[
                StudentTOutput(),
                NormalFixedScaleOutput(),
                NegativeBinomialOutput(),
                LogNormalOutput(),
            ]
        )

        # TODO: check the meaning of patch_sizes
        moirai_module_args = {
            "distr_output": mixture_output,
            "d_model": 384,
            "num_layers": 6,
            "patch_sizes": [1],  # tuple[int, ...] | list[int]
            "max_seq_len": 512,
            "attn_dropout_p": 0.0,
            "dropout_p": 0.0,
            "scaling": True,
        }

        self.module = MoiraiModule(**moirai_module_args)

    def forward(self, target, observed_mask, prediction_mask) -> Distribution:
        """
        see src/uni2ts/model/moirai/pretrain.py MoiraiPretrain.forward
        Parameters: (convert from ProbTS batch_data)
            target,  # Float[torch.Tensor, "*batch seq_len max_patch"]
            observed_mask,  # Bool[torch.Tensor, "*batch seq_len max_patch"],
            sample_id,  # Int[torch.Tensor, "*batch seq_len"],
            time_id,  # Int[torch.Tensor, "*batch seq_len"],
            variate_id,  # Int[torch.Tensor, "*batch seq_len"],
            prediction_mask,  # Bool[torch.Tensor, "*batch seq_len"],
            patch_size,  # Int[torch.Tensor, "*batch seq_len"],
        """
        # TODO: these are all fake values for now, need to be updated to real values
        sample_id = torch.zeros(
            target.shape[0], target.shape[1], dtype=torch.int64, device=target.device
        )
        time_id = torch.zeros(
            target.shape[0], target.shape[1], dtype=torch.int64, device=target.device
        )
        variate_id = torch.zeros(
            target.shape[0], target.shape[1], dtype=torch.int64, device=target.device
        )
        patch_size = torch.ones(
            target.shape[0], target.shape[1], dtype=torch.int64, device=target.device
        )
        distr = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return distr

    def forecast(self, batch_data, num_samples=None):
        target = rearrange(batch_data.past_target_cdf, 'b l k -> (b k) l 1')
        observed_mask = rearrange(batch_data.past_observed_values.bool(), 'b l k -> (b k) l 1')
        prediction_mask = rearrange(batch_data.future_observed_values.bool(), 'b l k -> (b k) l')

        pred_distr = self(target, observed_mask, prediction_mask)
        pred = pred_distr.sample(torch.Size((num_samples,)))
        return pred

    def get_weighted_loss(self, batch_data, loss):
        observed_values = batch_data.future_observed_values
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1)
        return loss

    def loss(self, batch_data, training_range=None):
        # pred_len = batch_data.future_target_cdf.shape[1]
        target = batch_data.past_target_cdf
        observed_mask = batch_data.past_observed_values.bool()
        prediction_mask = batch_data.future_observed_values.squeeze().bool()

        pred_distr = self(target, observed_mask, prediction_mask)
        loss = self._loss_func(pred_distr, batch_data.future_target_cdf)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def _loss_func(self, pred, target):
        """
        see src/uni2ts/loss/packed/distribution.py PackedNLLLoss
        Parameters:
            pred: Distribution,
            target: Float[torch.Tensor, "*batch seq_len #dim"],
            prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
            observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
            sample_id: Int[torch.Tensor, "*batch seq_len"],
            variate_id: Int[torch.Tensor, "*batch seq_len"],
        Returns:
            loss: Float[torch.Tensor, "*batch seq_len #dim"]
        """
        return -pred.log_prob(target)
