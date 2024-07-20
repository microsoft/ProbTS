# Checkpoints for Foundation Models

For full reproducibility, we provide the checkpoints for some foundation models as of the paper completion date. 

Download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1FaCk9Lj9KZGEO09gehNqC4fbTj4wnN8j?usp=sharing) with:
    
```bash
# By downloading, you agree to the terms of the original license agreements.
sh scripts/prepare_checkpoints.sh # in root directory
```


You can also download the newest checkpoints from the following repositories:

- For `Timer`, download the checkpoints from its [official repository](https://github.com/thuml/Large-Time-Series-Model?tab=readme-ov-file#code-for-fine-tuning) ([Google Drive](https://drive.google.com/drive/folders/15oaiAl4OO5gFqZMJD2lOtX2fxHbpgcU8) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/235e6bfcf5fa440bb119/)) under the folder `/path/to/checkpoints/timer/`.
- For `ForecastPFN`, download the checkpoints from its [official repository](https://github.com/abacusai/ForecastPFN#installation-) ([Google Drive](https://drive.google.com/file/d/1acp5thS7I4g_6Gw40wNFGnU1Sx14z0cU/view)) under the folder `/path/to/checkpoints/forecastpfn/`.
- For `UniTS`, download the checkpoints `units_x128_pretrain_checkpoint.pth` from its [official repository](https://github.com/mims-harvard/UniTS/releases/tag/ckpt) under the folder `/path/to/checkpoints/units/`.
- For `Lag-Llama`, download the checkpoints `lag-llama.ckpt` from its [huggingface repository](https://huggingface.co/time-series-foundation-models/Lag-Llama/tree/main) under the folder `/path/to/checkpoints/lag_llama/`.
- For other models, they can be automatically downloaded from huggingface during the first run.

<center>

| **Model** | **HuggingFace** |
| --- | --- |
| `MOIRAI` | [Link](https://huggingface.co/Salesforce/moirai-1.0-R-small) |
| `Chronos` | [Link](https://huggingface.co/amazon/chronos-t5-large) |
| `TinyTimeMixer` | [Link](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1) |
| `TimesFM` | [Link](https://huggingface.co/google/timesfm-1.0-200m) |

</center>

## For embedding mode:

You need to make the following change to `moirai` (`submodules/uni2ts/src/uni2ts/model/moirai/module.py`, line 140)
```python
embeddings = reprs.detach().clone()
distr_param = self.param_proj(reprs, patch_size)
distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
return distr, embeddings
```