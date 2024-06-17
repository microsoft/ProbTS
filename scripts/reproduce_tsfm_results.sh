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
MODEL='moirai'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf' 'electricity_ltsf'; do
    for CTX_LEN in 5000 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            for variate_mode in 'M' 'S'; do
                python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                    --data.data_manager.init_args.path ${DATA_DIR} \
                    --trainer.default_root_dir ${LOG_DIR} \
                    --data.data_manager.init_args.split_val true \
                    --data.data_manager.init_args.dataset ${DATASET} \
                    --data.data_manager.init_args.context_length ${CTX_LEN} \
                    --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                    --model.forecaster.init_args.variate_mode ${variate_mode} \
                    --data.test_batch_size 32
            done
        done
    done
done

for DATASET in 'exchange_rate_nips' 'solar_nips' 'electricity_nips'; do
    for CTX_LEN in 5000 96; do
        for PRED_LEN in 24; do
            for variate_mode in 'M' 'S'; do
                python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                    --data.data_manager.init_args.path ${DATA_DIR} \
                    --trainer.default_root_dir ${LOG_DIR} \
                    --data.data_manager.init_args.split_val true \
                    --data.data_manager.init_args.dataset ${DATASET} \
                    --data.data_manager.init_args.context_length ${CTX_LEN} \
                    --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                    --model.forecaster.init_args.variate_mode ${variate_mode} \
                    --data.test_batch_size 32
            done
        done
    done
done

# Chronos
MODEL='chronos'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf'; do
    for CTX_LEN in 5000 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 1
        done
    done
done

for DATASET in 'exchange_rate_nips' 'traffic_nips'; do
    for CTX_LEN in 512 96; do
        for PRED_LEN in 24; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 1
        done
    done
done

# Lag-Llama
MODEL='lag_llama'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf'; do
    for CTX_LEN in 512; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --model.forecaster.init_args.ckpt_path './checkpoints/lag-llama/lag-llama.ckpt' \
                --data.test_batch_size 1
        done
    done
done

# TimesFM
MODEL='timesfm'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2'; do
    for CTX_LEN in 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --data.test_batch_size 64
        done
    done
done

# Timer
MODEL='timer'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf' 'electricity_ltsf'; do
    for CTX_LEN in 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --model.forecaster.init_args.ckpt_path './checkpoints/timer/Timer_67M_UTSD_4G.pt' \
                --data.test_batch_size 64
        done
    done
done

# UniTS
MODEL='units'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2'; do
    for CTX_LEN in 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --model.forecaster.init_args.ckpt_path './checkpoints/units/units_x128_pretrain_checkpoint.pth' \
                --data.test_batch_size 64
        done
    done
done

# ForecastPFN
MODEL='forecastpfn'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf'; do
    for CTX_LEN in 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                --model.forecaster.init_args.ckpt_path './checkpoints/ForecastPFN/saved_weights' \
                --data.test_batch_size 64
        done
    done
done
