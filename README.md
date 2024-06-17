<div align=center> <img src="docs/figs/probts_logo.png" width = 50%/> </div>

# ProbTS: Benchmarking Point and Distributional Forecasting across Diverse Prediction Horizons

----------
[ [Paper](https://arxiv.org/abs/2310.07446) | [Benchmarking](./docs/benchmark/README.md) | [Documentation](./docs/documentation/README.md) ]

A wide range of industrial applications desire precise point and distributional forecasting for diverse prediction horizons. ProbTS serves as a benchmarking tool to aid in understanding how advanced time-series models fulfill these essential forecasting needs. It also sheds light on their advantages and disadvantages in addressing different challenges and unveil the possibilities for future research.

To achieve these objectives, ProbTS provides a unified pipeline that implements [cutting-edge models](#-available-models) from different research threads, including:
- Long-term point forecasting approaches, such as [PatchTST](https://arxiv.org/abs/2211.14730), [iTransformer](https://arxiv.org/abs/2310.06625), etc.
- Short-term probabilistic forecasting methods, such as [TimeGrad](https://arxiv.org/abs/2101.12072), [CSDI](https://arxiv.org/abs/2107.03502), etc.
- Recent time-series foundation models for universal forecasting, such as [TimesFM](https://arxiv.org/abs/2310.10688), [MOIRAI](https://arxiv.org/abs/2402.02592), etc.

Specifically, ProbTS emphasizes the differences in their primary methodological designs, including:
- Supporting point or distributional forecasts
- Using autoregressive or non-autoregressive decoding schemes for multi-step outputs

<div align=center> <img src="docs/figs/probts_framework.png" width = 95%/> </div>

## Available Models 🧩

ProbTS includes both classical time-series models, specializing in long-term point forecasting or short-term distributional forecasting, and recent time-series foundation models that offer zero-shot and arbitrary-horizon forecasting capabilities for new time series.

### Classical Time-series Models

| **Model** | **Original Eval. Horizon** | **Estimation** | **Decoding Scheme** | **Class Path** |
| --- | --- | --- | --- | --- |
| Linear | - | Point | Auto / Non-auto | `probts.model.forecaster.point_forecaster.LinearForecaster` |
| [GRU](https://arxiv.org/abs/1412.3555) | - | Point | Auto / Non-auto | `probts.model.forecaster.point_forecaster.GRUForecaster` |
| [Transformer](https://arxiv.org/abs/1706.03762) | - | Point | Auto / Non-auto | `probts.model.forecaster.point_forecaster.TransformerForecaster` |
| [Autoformer](https://arxiv.org/abs/2106.13008) | Long-trem | Point | Non-auto | `probts.model.forecaster.point_forecaster.Autoformer` |
| [N-HiTS](https://arxiv.org/abs/2201.12886) | Long-trem | Point | Non-auto | `probts.model.forecaster.point_forecaster.NHiTS` |
| [NLinear](https://arxiv.org/abs/2205.13504) | Long-trem | Point | Non-auto | `probts.model.forecaster.point_forecaster.NLinear` |
| [DLinear](https://arxiv.org/abs/2205.13504) | Long-trem | Point | Non-auto | `probts.model.forecaster.point_forecaster.DLinear` |
| [TimesNet](https://arxiv.org/abs/2210.02186) | Short- / Long-term | Point | Non-auto | `probts.model.forecaster.point_forecaster.TimesNet` |
| [PatchTST](https://arxiv.org/abs/2211.14730) | Long-trem | Point | Non-auto | `probts.model.forecaster.point_forecaster.PatchTST` |
| [iTransformer](https://arxiv.org/abs/2310.06625) | Long-trem | Point | Non-auto | `probts.model.forecaster.point_forecaster.iTransformer` |
| [GRU NVP](https://arxiv.org/abs/2002.06103) | Short-term | Probabilistic | Auto | `probts.model.forecaster.prob_forecaster.GRU_NVP` |
| [GRU MAF](https://arxiv.org/abs/2002.06103) | Short-term | Probabilistic | Auto | `probts.model.forecaster.prob_forecaster.GRU_MAF` |
| [Trans MAF](https://arxiv.org/abs/2002.06103) | Short-term | Probabilistic | Auto | `probts.model.forecaster.prob_forecaster.Trans_MAF` |
| [TimeGrad](https://arxiv.org/abs/2101.12072) | Short-term | Probabilistic | Auto | `probts.model.forecaster.prob_forecaster.TimeGrad` |
| [CSDI](https://arxiv.org/abs/2107.03502) | Short-term | Probabilistic | Non-auto | `probts.model.forecaster.prob_forecaster.CSDI` |
| [TSDiff](https://arxiv.org/abs/2307.11494) | Short-term | Probabilistic | Non-auto | `probts.model.forecaster.prob_forecaster.TSDiffCond` |

### Fundation Models

| **Model** | **Any Horizon** | **Estimation** | **Decoding Scheme** | **Class Path** |
| --- | --- | --- | --- | --- |
| [Lag-Llama](https://arxiv.org/abs/2310.08278) | &#x2714; | Probabilistic | Auto | `probts.model.forecaster.prob_forecaster.LagLlama` |
| [ForecastPFN](https://arxiv.org/abs/2311.01933) | &#x2714; | Point | Non-auto | `probts.model.forecaster.point_forecaster.ForecastPFN` |
| [TimesFM](https://arxiv.org/abs/2310.10688) | &#x2714; | Point | Auto | `probts.model.forecaster.point_forecaster.TimesFM` |
| [TTM](https://arxiv.org/abs/2401.03955) | &#x2718; | Point | Non-auto | `probts.model.forecaster.point_forecaster.TinyTimeMixer` |
| [Timer](https://arxiv.org/abs/2402.02368) | &#x2714; | Point | Auto | `probts.model.forecaster.point_forecaster.Timer` |
| [MOIRAI](https://arxiv.org/abs/2402.02592) | &#x2714; | Probabilistic | Non-auto | `probts.model.forecaster.prob_forecaster.Moirai` |
| [UniTS](https://arxiv.org/abs/2403.00131) | &#x2714; | Point | Non-auto | `probts.model.forecaster.point_forecaster.UniTS` |
| [Chronos](https://arxiv.org/abs/2403.07815) | &#x2714; | Probabilistic | Auto | `probts.model.forecaster.prob_forecaster.Chronos` |

Stay tuned for more models to be added in the future.


## Setup :wrench:

### Environment

ProbTS is developed with Python 3.10 and relies on [PyTorch Lightning](https://github.com/Lightning-AI/lightning). To set up the environment:

```bash
# Create a new conda environment
conda create -n probts python=3.10
conda activate probts

# Install required packages
pip install .
pip uninstall -y probts # recommended to uninstall the root package (optional)
```

[Optional] For time-series foundation models, you need to install basic packages and additional dependencies:

```bash
# Create a new conda environment
conda create -n probts_fm python=3.10
conda activate probts_fm

# Install required packages
pip install .

# Git submodule
git submodule update --init --recursive

# Install additional packages for foundation models
pip install ".[tsfm]"
pip uninstall -y probts # recommended to uninstall the root package (optional)

# For MOIRAI, we fix the version of the package for better performance
cd submodules/uni2ts
git reset --hard fce6a6f57bc3bc1a57c7feb3abc6c7eb2f264301
```

<details>

<summary>Optional for TSFMs reproducibility</summary>

```bash
# For TimesFM, fix the version for reproducibility (optional)
cd submodules/timesfm
git reset --hard 5c7b905

# For Lag-Llama, fix the version for reproducibility (optional)
cd submodules/lag_llama
git reset --hard 4ad82d9

# For TinyTimeMixer, fix the version for reproducibility (optional)
cd submodules/tsfm
git reset --hard bb125c14a05e4231636d6b64f8951d5fe96da1dc
```

</details>

### Datasets

- **Short-Term Forecasting**: We use datasets from [GluonTS](https://github.com/awslabs/gluonts). 
    Configure the datasets using `--data.data_manager.init_args.dataset {DATASET_NAME}`. You can choose from multivariate or univariate datasets as per your requirement.
    ```bash
    # Multivariate Datasets
    ['exchange_rate_nips', 'electricity_nips', 'traffic_nips', 'solar_nips', 'wiki2000_nips']

    # Univariate Datasets
    ['tourism_monthly', 'tourism_quarterly', 'tourism_yearly', 'm4_hourly', 'm4_daily', 'm4_weekly', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5']
    ```

- **Long-Term Forecasting**: To download the [long-term forecasting datasets](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy), please follow these steps:
    ```bash
    bash scripts/prepare_datasets.sh "./datasets"
    ```

    Configure the datasets using `--data.data_manager.init_args.dataset {DATASET_NAME}` with the following list of available datasets:
    ```bash
    # Long-term Forecasting
    ['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf', 'caiso', 'nordpool']
    ```
    Note: When utilizing long-term forecasting datasets, you must explicitly specify the `context_length` and `prediction_length` parameters. For example, to set a context length of 96 and a prediction length of 192, use the following command-line arguments:
    ```bash
    --data.data_manager.init_args.context_length 96 \
    --data.data_manager.init_args.prediction_length 192 \
    ```


### Checkpoints for Foundation Models

Download the checkpoints with the following command (details can be found [here](./checkpoints/README.md)):
```bash
bash scripts/prepare_checkpoints.sh # By downloading, you agree to the original licenses
```

## Quick Start :rocket:

Specify `--config` with a specific configuration file to reproduce results of point or probabilistic models on commonly used long- and short-term forecasting datasets. Configuration files are included in the [config](./config/) folder.

To run models:
```bash 
bash run.sh
```

Experimental results reproduction:

- **Long-term Forecasting:**

    ```bash 
    bash scripts/reproduce_ltsf_results.sh
    ```


- **Short-term Forecasting:**

    ```bash 
    bash scripts/reproduce_stsf_results.sh
    ```

- **Time Series Foundation Models:**

    ```bash 
    bash scripts/reproduce_tsfm_results.sh
    ```

### Short-term Forecasting Configuration

For short-term forecasting scenarios, datasets and corresponding `context_length` and `prediction_length` are automatically obtained from [GluonTS](https://github.com/awslabs/gluonts). Use the following command:

```bash 
python run.py --config config/path/to/model.yaml \
                --data.data_manager.init_args.path /path/to/datasets/ \
                --trainer.default_root_dir /path/to/log_dir/ \
                --data.data_manager.init_args.dataset {DATASET_NAME}
```
See full `DATASET_NAME` list:
```python
from gluonts.dataset.repository import dataset_names
print(dataset_names)
```

### Long-term Forecasting Configuration

For long-term forecasting scenarios, `context_length` and `prediction_length` must be explicitly assigned:

```bash 
python run.py --config config/path/to/model.yaml \
                --data.data_manager.init_args.path /path/to/datasets/ \
                --trainer.default_root_dir /path/to/log_dir/ \
                --data.data_manager.init_args.dataset {DATASET_NAME} \
                --data.data_manager.init_args.context_length {CTX_LEN} \
                --data.data_manager.init_args.prediction_length {PRED_LEN} 
```

`DATASET_NAME` options:
```bash 
['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf', 'caiso', 'nordpool']
```

## Benchmarking :balance_scale:

By utilizing ProbTS, we conduct a systematic comparison between studies that focus on point forecasting and those aimed at distributional estimation, employing various forecasting horizons and evaluation metrics. For more details

- [Short-term & Long-term Forecasting Benchmarking](./docs/benchmark/README.md)
- [Evaluating Time Series Foundation Models](./docs/benchmark/FOUNDATION_MODEL.md)


## Documentation :open_book:

For detailed information on configuration parameters and model customization, please refer to the [documentation](./docs/documentation/README.md).


### Key Configuration Parameters 

- Adjust model and data parameters in `run.sh`. Key parameters include:

| Config Name | Type | Description |
| --- | --- | --- |
| `trainer.max_epochs` | integer | Maximum number of training epochs. |
| `model.forecaster.class_path` | string | Forecaster module path (e.g., `probts.model.forecaster.point_forecaster.PatchTST`). |
| `model.forecaster.init_args.{ARG}` | - | Model-specific hyperparameters. |
| `model.num_samples` | integer | Number of samples per distribution during evaluation. |
| `model.learning_rate` | float | Learning rate. |
| `data.data_manager.init_args.dataset` | string | Dataset for training and evaluation. |
| `data.data_manager.init_args.path` | string | Path to the dataset folder. |
| `data.data_manager.init_args.scaler` | string | Scaler type: `identity`, `standard` (z-score normalization), or `temporal` (scale based on average temporal absolute value). |
| `data.data_manager.init_args.context_length` | integer | Length of observation window (required for long-term forecasting). |
| `data.data_manager.init_args.prediction_length` | integer | Forecasting horizon length (required for long-term forecasting). |
| `data.data_manager.init_args.var_specific_norm` | boolean | If conduct per-variate normalization or not. |
| `data.batch_size` | integer | Batch size. |


- To print the full pipeline configuration to a file:

    ```bash
    python run.py --print_config > config/pipeline_config.yaml
    ```

## Acknowledgement 🌟

Special thanks to the following repositories for their open-sourced code bases and datasets.

### Tools/Packages

- [GluonTS](https://github.com/awslabs/gluonts)
- [PyTorch-TS](https://github.com/zalandoresearch/pytorch-ts)
- [TSLib](https://github.com/libts/tslib) 
- [NeuralForecast](https://github.com/Nixtla/neuralforecast)

### Official Implementations

**Classical Time-series Models**

- [Autoformer](https://github.com/thuml/Autoformer)
- [N-HiTS](https://github.com/cchallu/n-hits)
- [NLinear, DLinear](https://github.com/cure-lab/LTSF-Linear)
- [TimesNet](https://github.com/thuml/Time-Series-Library)
- [RevIN](https://github.com/ts-kim/RevIN)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [iTransformer](https://github.com/thuml/iTransformer)
- [GRU NVP, GRU MAF, Trans MAF, TimeGrad](https://github.com/zalandoresearch/pytorch-ts/tree/master)
- [CSDI](https://github.com/ermongroup/CSDI)
- [TSDiff](https://github.com/amazon-science/unconditional-time-series-diffusion)


**Time-series Foundation Models**

- [MOIRAI](https://github.com/SalesforceAIResearch/uni2ts)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama)
- [TimesFM](https://github.com/google-research/timesfm)
- [Timer](https://github.com/thuml/Large-Time-Series-Model)
- [UniTS](https://github.com/mims-harvard/UniTS)
- [ForecastPFN](https://github.com/abacusai/ForecastPFN)
- [TTM](https://github.com/ibm-granite/granite-tsfm)

## Citing ProbTS 🌟

If you have used ProbTS for research or production, please cite it as follows.
```tex
@article{zhang2023probts,
  title={{ProbTS}: Benchmarking Point and Distributional Forecasting across Diverse Prediction Horizons},
  author={Zhang, Jiawen and Wen, Xumeng and Zhang, Zhenwei and Zheng, Shun and Li, Jia and Bian, Jiang},
  journal={arXiv preprint arXiv:2310.07446},
  year={2023}
}
```