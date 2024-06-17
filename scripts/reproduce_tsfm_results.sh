export CUDA_VISIBLE_DEVICES=0

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
# replace ${MODEL} with tarfet model name, e.g, GRU_NVP
# replace ${DATASET} with dataset name

# if not specify dataset_path, the default path is ./datasets

# MOIRAI
MODEL = 'moirai'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf' 'electricity_ltsf'
    for CTX_LEN in 5000 96
        for PRED_LEN in 24 48 96 192 336 720
            for variate_mode in 'M' 'S'
                python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                    --data.data_manager.init_args.path ${DATA_DIR} \
                    --trainer.default_root_dir ${LOG_DIR} \
                    --data.data_manager.init_args.split_val true \
                    --data.data_manager.init_args.dataset ${DATASET} \
                    --data.data_manager.init_args.context_length ${CTX_LEN} \
                    --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                    --model.encoder.init_args.variate_mode ${variate_mode} \
                    --data.test_batch_size 64 \

for DATASET in 'exchange_rate_nips' 'solar_nips' 'electricity_nips'
    for CTX_LEN in 5000 96
        for PRED_LEN in 24
            for variate_mode in 'M' 'S'
                python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                    --data.data_manager.init_args.path ${DATA_DIR} \
                    --trainer.default_root_dir ${LOG_DIR} \
                    --data.data_manager.init_args.split_val true \
                    --data.data_manager.init_args.dataset ${DATASET} \
                    --data.data_manager.init_args.context_length ${CTX_LEN} \
                    --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                    --model.encoder.init_args.variate_mode ${variate_mode} \
                    --data.test_batch_size 64 \

# Chronos
MODEL = 'chronos'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf'
    for CTX_LEN in 5000 96
        for PRED_LEN in 24 48 96 192 336 720
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 1 \

for DATASET in 'exchange_rate_nips' 'traffic_nips'
    for CTX_LEN in 512 96
        for PRED_LEN in 24
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 1 \

# Lag-Llama
MODEL = 'lag_llama'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf'
    for CTX_LEN in 512
        for PRED_LEN in 24 48 96 192 336 720
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 1 \

# TimesFM
MODEL = 'timesfm'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2'
    for CTX_LEN in 96
        for PRED_LEN in 24 48 96 192 336 720
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 64 \

# Timer
MODEL = 'timer'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf' 'electricity_ltsf'
    for CTX_LEN in 96
        for PRED_LEN in 24 48 96 192 336 720
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 64 \

# UniTS
MODEL = 'units'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2'
    for CTX_LEN in 96
        for PRED_LEN in 24 48 96 192 336 720
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 64 \

# ForecastPFN
MODEL = 'forecastpfn'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf'
    for CTX_LEN in 96
        for PRED_LEN in 24 48 96 192 336 720
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 64 \