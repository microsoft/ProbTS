# ---------------------------------------------------------------------------------
# Portions of this file are derived from uni2ts
# - Source: https://github.com/SalesforceAIResearch/uni2ts
# - Paper: Unified Training of Universal Time Series Forecasting Transformers
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import math
from typing import Optional, Union

import torch
from einops import rearrange, reduce, repeat

# from torch.distributions import Distribution
from uni2ts.distribution import (
    LogNormalOutput,
    MixtureOutput,
    NegativeBinomialOutput,
    NormalFixedScaleOutput,
    StudentTOutput,
)
from uni2ts.model.moirai.module import MoiraiModule

from probts.model.forecaster import Forecaster

# from probts.model.nn.layers.Moirai_backbone import MoiraiBackbone
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
        self.variate_mode = variate_mode
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
        self.moirai_module_args = {
            "distr_output": mixture_output,
            "d_model": 384,
            "num_layers": 6,
            "patch_sizes": [16],  # tuple[int, ...] | list[int]
            "max_seq_len": 512,
            "attn_dropout_p": 0.0,
            "dropout_p": 0.0,
            "scaling": True,
        }
        self.max_patch_size = max(self.moirai_module_args["patch_sizes"])

        # tmp set prediction length
        # prediction_length = 96
        self.module = MoiraiModule(**self.moirai_module_args)
        # self.moirai = MoiraiBackbone(
        #     context_length=self.context_length,
        #     target_dim=self.target_dim if self.variate_mode == "M" else 1,
        #     prediction_length=prediction_length,
        #     module=None,
        #     module_kwargs=self.moirai_module_args,
        #     scaling=scaling
        # )

    def forward(self, batch_data):
        B, L, K = batch_data.past_target_cdf.shape
        prediction_length = batch_data.future_target_cdf.shape[1]
        context_len = batch_data.past_target_cdf.shape[1]

        patch_size = self.moirai_module_args["patch_sizes"][0]
        past_target = rearrange(batch_data.past_target_cdf, 'b l k -> (b k) l 1')
        past_observed_target = rearrange(batch_data.past_observed_values, 'b l k -> (b k) l 1').int()
        past_is_pad = repeat(batch_data.past_is_pad, 'b l -> (b k) l', k=K).int()

        future_target = rearrange(batch_data.future_target_cdf, 'b l k -> (b k) l 1')
        # future_observed_target = rearrange(batch_data.future_observed_values, 'b l k -> (b k) l 1').int()

        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            patch_size,
            prediction_length,
            context_len,
            past_target,
            past_observed_target,
            past_is_pad,
            future_target=future_target,
        )

        distr = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=torch.ones_like(time_id, dtype=torch.long) * patch_size,
        )
        return distr, target

    def loss(self, batch_data, training_range=None):
        distr, target = self.forward(batch_data)
        loss = self._loss_func(distr, target)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        prediction_length = batch_data.future_target_cdf.shape[1]
        context_len = batch_data.past_target_cdf.shape[1]

        distr, _ = self.forward(batch_data)
        pred = distr.sample(torch.Size((num_samples,)))
        preds = self._format_preds(
            self.max_patch_size,
            pred,
            1,
            context_len,
            prediction_length,
        )
        return preds.unsqueeze(-1)

    def get_weighted_loss(self, batch_data, loss):
        observed_values = batch_data.future_observed_values
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1)
        return loss

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

    def _convert(
        self,
        patch_size: int,
        prediction_length: int,
        context_length: int,
        past_target,
        past_observed_target,
        past_is_pad,
        future_target=None,
        future_observed_target=None,
        future_is_pad=None,
        feat_dynamic_real=None,
        observed_feat_dynamic_real=None,
        past_feat_dynamic_real=None,
        past_observed_feat_dynamic_real=None,
    ):
        """
        Args:
            patch_size (int): Size of the patch.
            past_target (torch.Tensor): Shape [batch, past_time, tgt].
            past_observed_target (torch.Tensor): Shape [batch, past_time, tgt].
            past_is_pad (torch.Tensor): Shape [batch, past_time].
            future_target (Optional[torch.Tensor]): Shape [batch, future_time, tgt].
            future_observed_target (Optional[torch.Tensor]): Shape [batch, future_time, tgt].
            future_is_pad (Optional[torch.Tensor]): Shape [batch, future_time].
            feat_dynamic_real (Optional[torch.Tensor]): Shape [batch, time, feat].
            observed_feat_dynamic_real (Optional[torch.Tensor]): Shape [batch, time, feat].
            past_feat_dynamic_real (Optional[torch.Tensor]): Shape [batch, past_time, past_feat].
            past_observed_feat_dynamic_real (Optional[torch.Tensor]): Shape [batch, past_time, past_feat].
        Return:
            tuple[
                Float[torch.Tensor, "batch combine_seq patch"],  # target
                Bool[torch.Tensor, "batch combine_seq patch"],  # observed_mask
                Int[torch.Tensor, "batch combine_seq"],  # sample_id
                Int[torch.Tensor, "batch combine_seq"],  # time_id
                Int[torch.Tensor, "batch combine_seq"],  # variate_id
                Bool[torch.Tensor, "batch combine_seq"],  # prediction_mask
            ]:
        """

        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, prediction_length, past_observed_target
        )

        # ---- added by zhenwei ----
        context_token_length = math.ceil(context_length / patch_size)
        prediction_token_length = math.ceil(prediction_length / patch_size)
        # ---- added by zhenwei ----

        if future_target is None:
            future_target = torch.zeros(
                batch_shape
                + (
                    prediction_length,
                    past_target.shape[-1],
                ),
                dtype=past_target.dtype,
                device=device,
            )
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (
                    prediction_length,
                    past_observed_target.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=context_token_length,
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=prediction_token_length,
                ),
            ]
        )
        dim_count += past_target.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (context_token_length * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        prediction_token_length
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        if feat_dynamic_real is not None:
            if observed_feat_dynamic_real is None:
                raise ValueError(
                    "observed_feat_dynamic_real must be provided if feat_dynamic_real is provided"
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., : context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )
            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., : context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )
            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=feat_dynamic_real.shape[-1],
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            prediction_token_length
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            time_id.extend(
                [past_seq_id] * feat_dynamic_real.shape[-1]
                + [future_seq_id] * feat_dynamic_real.shape[-1]
            )
            variate_id.extend(
                [
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=context_token_length,
                    ),
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=prediction_token_length,
                    ),
                ]
            )
            dim_count += feat_dynamic_real.shape[-1]
            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            context_token_length
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        batch_shape
                        + (
                            prediction_token_length
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

        if past_feat_dynamic_real is not None:
            if past_observed_feat_dynamic_real is None:
                raise ValueError(
                    "past_observed_feat_dynamic_real must be provided if past_feat_dynamic_real is provided"
                )
            target.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                )
            )
            observed_mask.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                )
            )
            sample_id.append(
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_feat_dynamic_real.shape[-1],
                )
            )
            time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])

            variate_id.append(
                repeat(
                    torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=context_token_length,
                )
            )
            dim_count += past_feat_dynamic_real.shape[-1]
            prediction_mask.append(
                torch.zeros(
                    batch_shape
                    + (
                        context_token_length
                        * past_feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

    def _generate_time_id(
        self,
        patch_size: int,
        prediction_length: int,
        past_observed_target, # : Bool[torch.Tensor, "batch past_seq tgt"]
    ):
        """
        -> tuple[
            Int[torch.Tensor, "batch past_token"], Int[torch.Tensor, "batch future_token"]
        ]:
        """
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(past_seq_id.cumsum(dim=-1) - 1, min=0)
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        # ---- added by zhenwei ----
        prediction_token_length = math.ceil(prediction_length / patch_size)
        # ---- added by zhenwei ----
        future_seq_id = (
            repeat(
                torch.arange(
                    prediction_token_length,
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)
    
    def _format_preds(
        self,
        patch_size: int,
        preds, # : Float[torch.Tensor, "sample batch combine_seq patch"],
        target_dim: int,
        context_length: int,
        prediction_length: int,
    ):
        # -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        context_token_length = math.ceil(context_length / patch_size)
        prediction_token_length = math.ceil(prediction_length / patch_size)

        start = target_dim * context_token_length
        end = start + target_dim * prediction_token_length
        preds = preds[..., start:end, :patch_size]
        preds = rearrange(
            preds,
            "sample ... (dim seq) patch -> ... sample (seq patch) dim",
            dim=target_dim,
        )[..., : prediction_length, :]
        return preds.squeeze(-1)