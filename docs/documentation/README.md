# Documentation :open_book:


## Configuration Parameters 

- To print the full pipeline configuration to a file:

    ```bash
    python run.py --print_config > config/pipeline_config.yaml
    ```

### Trainer

| Config Name | Type | Description |
| --- | --- | --- |
| `trainer.max_epochs` | `int` | Maximum number of training epochs. |
| `trainer.limit_train_batches` | `int` | Limits the number of training batches per epoch. |
| `trainer.check_val_every_n_epoch` | `int` | Perform validation every n training epochs. |
| `trainer.default_root_dir` | `int` | Default path for logs and weights. |
| `trainer.accumulate_grad_batches` | `int` | Number of batches to accumulate gradients before updating. |

### Model

| Config Name | Type | Description |
| --- | --- | --- |
| `model.forecaster.class_path` | `str` | Forecaster module path (e.g., `probts.model.forecaster.point_forecaster.PatchTST`). |
| `model.forecaster.init_args.{ARG}` | - | Model-specific hyperparameters. |
| `model.num_samples` | `int` | Number of samples per distribution during evaluation. |
| `model.learning_rate` | `float` | Learning rate. |
| `model.quantiles_num` | `int` | Number of quantiles for evaluation. |
| `model.sampling_weight_scheme` | `str`  | The scheme of training horizon reweighting. Options: ['random', 'none', 'const'].|

### Data

| Config Name | Type | Description |
| --- | --- | --- |
| `data.data_manager.init_args.dataset` | `str` | Dataset for training and evaluation. |
| `data.data_manager.init_args.path` | `str` | Path to the dataset folder. |
| `data.data_manager.init_args.split_val` | `bool` | Whether to split a validation set during training. |
| `data.data_manager.init_args.scaler` | `str` | Scaler type: `identity`, `standard` (z-score normalization), or `temporal` (scale based on average temporal absolute value). |
| `data.data_manager.init_args.target_dim` | `int` | The number of variates. |
| `data.data_manager.init_args.var_specific_norm` | `bool` | If conduct per-variate normalization or not. |
| `data.data_manager.init_args.timeenc` | `int` | Time feature type. Select from `[0,1,2]`. See the explaination below for details. |
| `data.data_manager.init_args.context_length`    | `Union[str, int, list]`       | Length of observation window in inference phase. |
| `data.data_manager.init_args.prediction_length` | `Union[str, int, list]`       | Forecasting horizon length in inference phase. |
| `data.data_manager.init_args.val_pred_len_list` | `Union[str, int, list]`       | Forecasting horizon length for performance validation. |
| `data.data_manager.init_args.val_ctx_len`       | `Union[str, int, list]`      | Forecasting horizons for performance validation. |
| `data.data_manager.init_args.train_pred_len_list`| `Union[str, int, list]`      | Length of observation window in training phase. |
| `data.data_manager.init_args.train_ctx_len` | `Union[str, int, list]`      | Forecasting horizons in training phase. |
| `data.data_manager.init_args.continuous_sample`  | `bool`   | If True, sampling horizons from `[min(train_pred_len_list), max(train_pred_len_list)]`, else sampling within the set `train_pred_len_list`.|
| `data.batch_size` | `int` | Batch size. |

#### Temporal Features

For the datasets used for long-term forecasting scenario, we support three types of time feature encoding

```bash
--data.data_manager.init_args.timeenc {the encoding type} # select from [0,1,2]
```

- **[timeenc 0] temporal information**

    The dimension of time feature is 5, containing `month, day, weekday, hour, minute`.

- **[timeenc 1] time feature based on frequency**
    Extract time feature using `time_features_from_frequency_str()` function. The dimensionality follows:
    ```bash
    freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
    ```

    *Note: timeenc = 0 if model.embed != 'timeF' else 1.*

- **[timeenc 2] Raw date information**

    The dimension of time feature is 5, using the following code to recover it to date data type:
    ```bash
    data_stamp = batch_data.past_time_feat.cpu().numpy().astype('datetime64[s]')
    data_stamp = batch_data.future_time_feat.cpu().numpy().astype('datetime64[s]')
    ```


## Customized Model

With our platform, you can easily evaluate customized models across various datasets. Follow the steps below to create and evaluate your model.


### Step 1: Create a New Python File

Create a new Python file and follow the structure below to define your custom model:

```python
from probts.model.forecaster import Forecaster

class ModelName(Forecaster):
    def __init__(
        self,
        **kwargs
    ):
        """
        Initialize the model with parameters.
        """
        super().__init__(**kwargs)
        # Initialize model parameters here

    def forward(self, inputs):
        """
        Forward pass for the model.

        Parameters:
        inputs [Tensor]: Input tensor for the model.

        Returns:
        Tensor: Output tensor.
        """
        # Perform the forward pass of the model
        return outputs

    def loss(self, batch_data):
        """
        Compute the loss for the given batch data.

        Parameters:
        batch_data [dict]: Dictionary containing input data and possibly target data.

        Returns:
        Tensor: Computed loss.
        """
        # Extract inputs and targets from batch_data
        inputs = batch_data.past_target_cdf[:, -self.context_length:, :] # [batch_size, context_length, var_num]
        target = batch_data.future_target_cdf # [batch_size, prediction_length, var_num]

        # Forward pass
        outputs = self.forward(inputs)
        
        # Calculate loss using a loss function, e.g., Mean Squared Error
        loss = self.loss_function(outputs, future_target_cdf)

        return loss

    def forecast(self, batch_data, num_samples=None):
        """
        Generate forecasts for the given batch data.

        Parameters:
        batch_data [dict]: Dictionary containing input data.
        num_samples [int, optional]: Number of samples per distribution during evaluation. Defaults to None.

        Returns:
        Tensor: Forecasted outputs.
        """
        # Perform the forward pass to get the outputs
        outputs = self(batch_data.past_target_cdf[:, -self.context_length:, :])

        if num_samples is not None:
            # If num_samples is specified, use it to sample from the distribution
            outputs = self.sample_from_distribution(outputs, num_samples)
        else: 
            # If perform point estimation, the num_samples is equal to 1
            outputs = outputs.unsqueeze(1)
        return outputs # [batch_size, num_samples, prediction_length, var_num]
```

#### Input Data Format

The `batch_data` dictionary contains several fields that provide necessary information for the model's operation. Each field is described below:

- **`target_dimension_indicator`**: 
  - **Shape**: [var_num]
  - **Description**: Indicator that specifies which dimension or feature of the target is being referenced. 

- **`{past|future}_time_feat`**: 
  - **Shape**: [batch_size,length,time_feature_dim]
  - **Description**: Time features associated with each time step in the past or future. This can include various time-related information such as timestamps, seasonal indicators (e.g., month, day of the week), or other temporal features that provide context to the observations.
- **`{past|future}_target_cdf`**: 
  - **Shape**: [batch_size,length,var_num]
  - **Description**: The observation values of the target variable(s) for past or future time steps. 
- **`{past|future}_observed_values`**: 
  - **Shape**: [batch_size,length,var_num]
  - **Description**: Binary masks indicating which values in the past or future target data are observed (1) and which are missing or unobserved (0). 

### Step 2: Create YAML Configuration File

Create a YAML configuration file (`model.yaml`) for the customized model:

```yaml
seed_everything: 1 # random seed
trainer:
  accelerator: gpu
  devices: 1
  strategy: auto
  max_epochs: 50
  use_distributed_sampler: false
  limit_train_batches: 100
  log_every_n_steps: 1
  default_root_dir: ./results # path to the log folder
model:
  forecaster:
    class_path: class.path.to.ModelName
    init_args:
      # init your hyperparameter here
  learning_rate: 0.001 # learning rate
data:
  data_manager:
    class_path: probts.data.data_manager.DataManager
    init_args:
      dataset: solar_nips # dataset name
      split_val: true
      scaler: standard # identity, standard, temporal
  batch_size: 32
  test_batch_size: 32
  num_workers: 8
```

### Step 3: Run the Customized Model

Run the customized model using the configuration file:

```bash
python run.py --config config/path/to/model.yaml
```


## Forecasting with Varied Prediction Lengths


**Example:**
```bash 
python run.py --config config/multi_hor/elastst.yaml \
                --data.data_manager.init_args.path ./datasets \
                --trainer.default_root_dir /path/to/log_dir/ \
                --data.data_manager.init_args.dataset {DATASET_NAME} \
                --data.data_manager.init_args.context_length ${TEST_CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${TEST_PRED_LEN} \
                --data.data_manager.init_args.train_ctx_len ${TRAIN_CTX_LEN} \
                --data.data_manager.init_args.train_pred_len_list ${TRAIN_PRED_LEN} \
                --data.data_manager.init_args.val_ctx_len ${VAL_CTX_LEN} \
                --data.data_manager.init_args.val_pred_len_list ${VAL_PRED_LEN} 
```

- `DATASET_NAME`: Select from datasets used in long-term forecasting scenerios.
- `TEST_CTX_LEN`: Context length in the testing phase.
- `VAL_CTX_LEN` (Default: `TEST_CTX_LEN`): Context length in the validation phase.
- `TRAIN_CTX_LEN` (Default: `TEST_CTX_LEN`): Context length in the training phase.
- `TEST_PRED_LEN`: Forecasting horizons in the testing phase.
- `VAL_PRED_LEN` (Default: `TEST_PRED_LEN`): Forecasting horizons for performance validation.
- `TRAIN_PRED_LEN` (Default: `TEST_PRED_LEN`): Forecasting horizons in the training phase.

The results across multiple horizons will be saved to: 
```bash 
/path/to/log_dir/{DATASET_NAME}_{MODEL}_{seed}_TrainCTX_{TRAIN_CTX_LEN}_TrainPRED_{TRAIN_PRED_LEN}_ValCTX_{CTX_LEN}_ValPRED_{VAL_PRED_LEN}/horizons_results.csv
```

### Example 1: Varied-Horizon Training

**Mode 1: Random sampling from a set of horizons**

```bash 
python run.py --config config/multi_hor/elastst.yaml \
                --data.data_manager.init_args.path ./datasets \
                --trainer.default_root_dir /path/to/log_dir/ \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length 96 \
                --data.data_manager.init_args.prediction_length 720 \
                --data.data_manager.init_args.train_ctx_len 96 \
                --data.data_manager.init_args.val_pred_len_list 720 \
                # random selection from {96, 192, 336, 720}
                --data.data_manager.init_args.train_pred_len_list 96-192-336-720 \
                --data.data_manager.init_args.continuous_sample false 
```

**Mode 2: Random sampling from a horizon range**

```bash 
python run.py --config config/multi_hor/elastst.yaml \
                --data.data_manager.init_args.path ./datasets \
                --trainer.default_root_dir /path/to/log_dir/ \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length 96 \
                --data.data_manager.init_args.prediction_length 720 \
                --data.data_manager.init_args.train_ctx_len 96 \
                --data.data_manager.init_args.val_pred_len_list 720 \
                # random sampling from [1, 720]
                --data.data_manager.init_args.train_pred_len_list 1-720 \ 
                --data.data_manager.init_args.continuous_sample true 
```

### Example 2: Validation and Testing with Multiple Horizons

```bash 
python run.py --config config/multi_hor/elastst.yaml \
                --data.data_manager.init_args.path ./datasets \
                --trainer.default_root_dir /path/to/log_dir/ \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length 96 \
                --data.data_manager.init_args.train_pred_len_list 720 \ 
                --data.data_manager.init_args.train_ctx_len 96 \
                # validation on {96, 192, 336, 720}
                --data.data_manager.init_args.val_pred_len_list 96-192-336-720 \
                # testing on {24, 96, 192, 336, 720, 1024}
                --data.data_manager.init_args.prediction_length 24-96-192-336-720-1024 
```
