DATA_DIR=/path/to/datasets
LOG_DIR=/path/to/log_dir

CTX_LEN=96

TRAIN_PRED_LEN=720
VAL_PRED_LEN=720
TEST_PRED_LEN=720

# select from ['etth1', 'etth2', 'ettm1', 'ettm2', 'traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'weather_ltsf']
DATASET='exchange_ltsf' 

# select from ['elastst', 'autoformer', 'dlinear', 'patchtst', 'transformer_enc', 'tsmixer']
MODEL=tsmixer

python run.py --config config/baselines/${DATASET}/${MODEL}.yaml --seed_everything 0  \
    --data.data_manager.init_args.path ${DATA_DIR} \
    --trainer.default_root_dir ${LOG_DIR} \
    --data.data_manager.init_args.split_val true \
    --data.data_manager.init_args.dataset ${DATASET} \
    --data.data_manager.init_args.context_length ${CTX_LEN} \
    --data.data_manager.init_args.prediction_length ${TEST_PRED_LEN} \
    --data.data_manager.init_args.train_pred_len_list ${TRAIN_PRED_LEN} \
    --data.data_manager.init_args.train_ctx_len_list ${CTX_LEN} \
    --data.data_manager.init_args.val_pred_len_list $VAL_PRED_LEN \
    --trainer.max_epochs 1 
