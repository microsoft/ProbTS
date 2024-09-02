
export CUDA_VISIBLE_DEVICES=0
MODEL=patchtst
DATASET="etth2"
CTX_LEN=96
PRED_LEN=96

DATA_DIR=./datasets
LOG_DIR=./exps

# multivariate datasets:
# ['exchange_rate_nips', 'solar_nips','electricity_nips', 'traffic_nips','wiki2000_nips']

# Univariate datasets:
# ['m4_weekly', 'm4_hourly', 'm4_daily', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5', 'tourism_monthly', 'tourism_quarterly', 'tourism_yearly']

# Long-term forecasting:
# ['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf']
# NOTE: when using long-term forecasting datasets, please explicit assign context_length and prediction_length, e.g., :
# --data.data_manager.init_args.context_length 96 \
# --data.data_manager.init_args.prediction_length 192 \

# run pipeline with train and test
# replace ${MODEL} with tarfet model name, e.g, patchtst
# replace ${DATASET} with dataset name

# if not specify dataset_path, the default path is ./datasets

python run.py --config config/ltsf/etth1/${MODEL}.yaml --seed_everything 0  \
    --data.data_manager.init_args.path ${DATA_DIR} \
    --trainer.default_root_dir ${LOG_DIR} \
    --data.data_manager.init_args.dataset monash_electricity_hourly \
    --data.data_manager.init_args.data_path /data/Blob_WestJP/v-jiawezhang/data/all_datasets/monash/solar_10_minutes_dataset.tsf \
    --data.data_manager.init_args.freq min \
    --data.data_manager.init_args.context_length 96 \
    --data.data_manager.init_args.prediction_length 96 \
    # --data.data_manager.init_args.multivariate true