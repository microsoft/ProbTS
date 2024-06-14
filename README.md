<div align=center> <img src="doc/figs/ProbTS.png" width = 50%/> </div>

# ProbTS: Benchmarking Point and Distributional Forecasting across Diverse Prediction Horizons

----------
[ [Paper](https://arxiv.org/abs/2310.07446) | [Benchmarking](./doc/benchmark/README.md) | [Documentation](./doc/documentation/README.md) ]

ProbTS is a benchmark tool designed as a unified platform to evaluate point and distributional forecasts across short and long horizons and to conduct a rigorous comparative analysis of numerous cutting-edge studies.

## Setup :wrench:

### Environment (TO BE UPDATE)

ProbTS is developed with Python 3.8 and relies on [PyTorch Lightning](https://github.com/Lightning-AI/lightning). To set up the environment:

```bash
# Install required packages
pip install -r requirements.txt
# Install uni2ts package
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
git reset --hard fce6a6f57bc3bc1a57c7feb3abc6c7eb2f264301
pip install -e '.[notebook]'
```

### Datasets

- **Long-Term Forecasting**: 
    To set up the long-term forecasting datasets, please follow these steps:
    1. Download long-term forecasting datasets from [HERE](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place them in `./dataset`. 
    2. Update the dataset configuration in `run.sh` with the following list of available datasets for long-term forecasting:
    ```bash
    # Long-term Forecasting
    ['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf']
    ```
    Note: When utilizing long-term forecasting datasets, you must explicitly specify the `context_length` and `prediction_length` parameters. For example, to set a context length of 96 and a prediction length of 192, use the following command-line arguments:
    ```bash
    --data.data_manager.init_args.context_length 96 \
    --data.data_manager.init_args.prediction_length 192 \
    ```
    3. [Opt.] Download CAISO and NordPool datasets from [DEPTS](https://github.com/weifantt/DEPTS/tree/main) and place them in ```./dataset```. Use the following dataset names in the configuration file:
    ```bash
    ['caiso', 'nordpool']
    ```


- **Short-Term Forecasting**: Use datasets from [GluonTS](https://github.com/awslabs/gluonts). 
    Configure the datasets for your forecasting model. You can choose from multivariate or univariate datasets as per your requirement.
    ```bash
    # Multivariate Datasets
    ['exchange_rate_nips', 'electricity_nips', 'traffic_nips', 'solar_nips', 'wiki2000_nips']

    # Univariate Datasets
    ['tourism_monthly', 'tourism_quarterly', 'tourism_yearly', 'm4_hourly', 'm4_daily', 'm4_weekly', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5']
    ```

## Quick Start :rocket:

Configure and run models using `run.sh`. Replace `--config` with the appropriate configuration file from the [config](./config/) folder.

```bash 
bash run.sh
```

## Available Models 🧩

ProbTS incorporates a range of models,ach with specific estimation paradigm and decoding schemes.  

### Non-universal Time-series Models

| **Model** | **Original Eval. Horizon** | **Estimation** | **Decoding Scheme** |
| --- | --- | --- | --- |
| Linear | - | Point | Auto / Non-auto |
| GRU | - | Point | Auto / Non-auto |
| Transformer | - | Point | Auto / Non-auto |
| [Autoformer (Wu et al., 2021)](https://github.com/thuml/Autoformer) | Long-trem | Point | Non-auto |
| [N-HiTS (Challu et al., 2023)](https://github.com/cchallu/n-hits) | Long-trem | Point | Non-auto |
| [NLinear (Zeng et al., 2023)](https://github.com/cure-lab/LTSF-Linear) | Long-trem | Point | Non-auto |
| [DLinear (Zeng et al., 2023)](https://github.com/cure-lab/LTSF-Linear) | Long-trem | Point | Non-auto |
| [PatchTST (Nie et al., 2023)](https://github.com/yuqinie98/PatchTST) | Long-trem | Point | Non-auto |
| [TimesNet (Wu et al., 2023)](https://github.com/thuml/Time-Series-Library) | Short- / Long-term | Point | Non-auto |
| [iTransformer (Liu et al., 2024)](https://github.com/thuml/iTransformer) | Long-trem | Point | Non-auto |
| [GRU NVP (Rasul et al., 2021)](https://github.com/zalandoresearch/pytorch-ts/tree/master) | Short-term | Probabilistic | Auto |
| [GRU MAF (Rasul et al., 2021)](https://github.com/zalandoresearch/pytorch-ts/tree/master) | Short-term | Probabilistic | Auto |
| [Trans MAF (Rasul et al., 2021)](https://github.com/zalandoresearch/pytorch-ts/tree/master) | Short-term | Probabilistic | Auto |
| [TimeGrad (Rasul et al., 2021)](https://github.com/zalandoresearch/pytorch-ts/tree/master) | Short-term | Probabilistic | Auto |
| [CSDI (Tashiro et al., 2021)](https://github.com/ermongroup/CSDI) | Short-term | Probabilistic | Non-auto |
| [TSDiff (Kollovieh et al., 2023)](https://github.com/amazon-science/unconditional-time-series-diffusion) | Short-term | Probabilistic | Non-auto |

Stay tuned for more models to be added in the future.

### Fundation Models (TO BE UPDATE)



## Benchmarking :balance_scale:

(TO BE UPDATE)

By utilizing ProbTS, we conduct a systematic comparison between studies that focus on point forecasting and those aimed at distributional estimation, employing various forecasting horizons and evaluation metrics. For more details

- [Long-term Forecasting Benchmarking](./doc/benchmark/long_term/)
- [Short-term Forecasting Benchmarking](./doc/benchmark/short_term/)
- [Evaluating Time Series Foundation Models](./doc/benchmark/foundation_model/)


## Documentation :open_book:

(TO BE UPDATE)

Detailed documentation see [HERE](./doc/documentation/README.md).


### Configuration Parameters 

- Adjust model and data parameters in `run.sh`. Key parameters include:

| Config Name | Type | Description |
| --- | --- | --- |
| `model.encoder.class_path` | string | Encoder module path (e.g., `probts.model.encoder.RNNEncoder`) |
| `model.forecaster.class_path` | string | Forecaster module path (e.g., `probts.model.forecaster.prob_forecaster.GaussianDiffusion`) |
| `model.num_samples` | integer | Number of samples per distribution during evaluation |
| `model.learning_rate` | float | Learning rate |
| `model.quantiles_num` | integer | Number of quantiles for evaluation |
| `data.data_manager.init_args.dataset` | string | Dataset for training and evaluation |
| `data.data_manager.init_args.path` | string | Path to the dataset folder |
| `data.data_manager.init_args.split_val` | boolean | Whether to split a validation set during training |
| `data.data_manager.init_args.scaler` | string | Scaler type: `none`, `standard` (z-score normalization), or `temporal` (scale based on average temporal absolute value) |
| `data.data_manager.init_args.context_length` | integer | Length of observation window (required for long-term forecasting) |
| `data.data_manager.init_args.prediction_length` | integer | Forecasting horizon length (required for long-term forecasting) |
| `data.data_manager.init_args.var_specific_norm` | boolean | If conduct per-variate normalization or not |
| `data.batch_size` | integer | Batch size |

- To print the full pipeline configuration to a file:

    ```bash
    python setup.py --print_config > config/pipeline_config.yaml
    ```



## Acknowledgement 🌟

Special thanks to the following repositories for their invaluable code bases and datasets:

- [GluonTS](https://github.com/awslabs/gluonts)
- [PyTorch-TS](https://github.com/zalandoresearch/pytorch-ts)
- [TSLib](https://github.com/libts/tslib) 
- [NeuralForecast](https://github.com/Nixtla/neuralforecast)
- [PatchTST](https://github.com/yuqinie98/PatchTST/tree/main)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
- [RevIN](https://github.com/ts-kim/RevIN)


