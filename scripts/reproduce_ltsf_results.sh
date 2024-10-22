export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps


CTX_LEN=96

for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 'weather_ltsf' 'electricity_ltsf' 'exchange_ltsf' 'traffic_ltsf'
do
    for MODEL in 'dlinear' 'patchtst' 'gru_nvp' 'timegrad' 'csdi'
    do
        for PRED_LEN in 96 192 336 720
        do
            python run.py --config config/ltsf/${DATASET}/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} 
        done
    done
done

CTX_LEN=36

for DATASET in 'illness_ltsf'
do
    for MODEL in 'dlinear' 'patchtst' 'gru_nvp' 'timegrad' 'csdi'
    do
        for PRED_LEN in 24 36 48 60
        do
            python run.py --config config/ltsf/${DATASET}/${MODEL}.yaml --seed_everything 0  \
                --data.data_manager.init_args.path ${DATA_DIR} \
                --trainer.default_root_dir ${LOG_DIR} \
                --data.data_manager.init_args.split_val true \
                --data.data_manager.init_args.dataset ${DATASET} \
                --data.data_manager.init_args.context_length ${CTX_LEN} \
                --data.data_manager.init_args.prediction_length ${PRED_LEN} 
        done
    done
done