export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps

# MOIRAI
MODEL='moirai'
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf' 'electricity_ltsf'; do
    for CTX_LEN in 5000 96; do
        for PRED_LEN in 24 48 96 192 336 720; do
            python run.py --config config/tsfm/${MODEL}/context_${CTX_LEN}/${DATASET}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN}
        done
    done
done

for DATASET in 'exchange_rate_nips' 'solar_nips' 'electricity_nips'; do
    for CTX_LEN in 5000 96; do
        python run.py --config config/tsfm/${MODEL}/context_${CTX_LEN}/${DATASET}.yaml --seed_everything 0  \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --trainer.default_root_dir ${LOG_DIR} \
            --data.data_manager.init_args.dataset ${DATASET} 
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