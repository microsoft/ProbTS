export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps/moirai_ori_pretrain
CTX_LEN=96
PRED_LEN=96

for DATASET in etth1 # etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
do
    python run.py --config config/tsfm/moirai.yaml \
        --seed_everything 0 \
        --trainer.default_root_dir "${LOG_DIR}/gluonts" \
        --data.data_manager.init_args.path ${DATA_DIR} \
        --data.data_manager.init_args.split_val true \
        --data.data_manager.init_args.context_length ${CTX_LEN} \
        --data.data_manager.init_args.prediction_length ${PRED_LEN} \
        --data.data_manager.init_args.dataset=${DATASET} \
        --model.forecaster.init_args.ckpt_path "/data/Blob_EastUS/v-zhenwzhang/log/moirai_pretrain/moirai_small/gluonts/gluonts/checkpoints/epoch=999-step=100000.ckpt" \
        --data.test_batch_size 1
done