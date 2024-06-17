# Documentation :open_book:


## Configuration Parameters 

- To print the full pipeline configuration to a file:

    ```bash
    python run.py --print_config > config/pipeline_config.yaml
    ```

### Trainer

| Config Name | Type | Description |
| --- | --- | --- |
| `trainer.max_epochs` | integer | Maximum number of training epochs. |
| `trainer.limit_train_batches` | integer | Limits the number of training batches per epoch. |
| `trainer.check_val_every_n_epoch` | integer | Perform validation every n training epochs. |
| `trainer.default_root_dir` | integer | Default path for logs and weights. |
| `trainer.accumulate_grad_batches` | integer | Number of batches to accumulate gradients before updating. |

### Model

| Config Name | Type | Description |
| --- | --- | --- |
| `model.forecaster.class_path` | string | Forecaster module path (e.g., `probts.model.forecaster.point_forecaster.PatchTST`). |
| `model.forecaster.init_args.{ARG}` | - | Model-specific hyperparameters. |
| `model.num_samples` | integer | Number of samples per distribution during evaluation. |
| `model.learning_rate` | float | Learning rate. |
| `model.quantiles_num` | integer | Number of quantiles for evaluation. |


### Data

| Config Name | Type | Description |
| --- | --- | --- |
| `data.data_manager.init_args.dataset` | string | Dataset for training and evaluation. |
| `data.data_manager.init_args.path` | string | Path to the dataset folder. |
| `data.data_manager.init_args.split_val` | boolean | Whether to split a validation set during training. |
| `data.data_manager.init_args.scaler` | string | Scaler type: `identity`, `standard` (z-score normalization), or `temporal` (scale based on average temporal absolute value). |
| `data.data_manager.init_args.context_length` | integer | Length of observation window (required for long-term forecasting). |
| `data.data_manager.init_args.prediction_length` | integer | Forecasting horizon length (required for long-term forecasting). |
| `data.data_manager.init_args.target_dim` | integer | The number of variates. |
| `data.data_manager.init_args.var_specific_norm` | boolean | If conduct per-variate normalization or not. |
| `data.data_manager.init_args.timeenc` | integer | Time feature type. Select from `[0,1,2]`. See the explaination below for details. |
| `data.batch_size` | integer | Batch size. |

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

    Note: timeenc = 0 if model.embed != 'timeF' else 1.

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