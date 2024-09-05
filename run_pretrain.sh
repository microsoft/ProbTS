export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps/moirai_ori_pretrain
CTX_LEN=96
PRED_LEN=96

for DATASET in etth1 etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
do
    # for PRE_DATA in buildings_bench cloudops_tsf largest proenfo subseasonal # cmip6 era5
    # do
    #     python run.py --config config/tsfm/moirai.yaml \
    #         --seed_everything 0 \
    #         --trainer.default_root_dir "${LOG_DIR}/${PRE_DATA}" \
    #         --data.data_manager.init_args.path ${DATA_DIR} \
    #         --data.data_manager.init_args.split_val true \
    #         --data.data_manager.init_args.context_length ${CTX_LEN} \
    #         --data.data_manager.init_args.prediction_length ${PRED_LEN} \
    #         --data.data_manager.init_args.dataset=${DATASET} \
    #         --model.forecaster.init_args.ckpt_path "/data/Blob_EastUS/v-zhenwzhang/log/moirai_pretrain/moirai_small/${PRE_DATA}/${PRE_DATA}/checkpoints/last-v1.ckpt" \
    #         --data.test_batch_size 1
    # done
    # for PRE_DATA in buildings_900k gluonts # cmip6 era5
    # do
    #     python run.py --config config/tsfm/moirai.yaml \
    #         --seed_everything 0 \
    #         --trainer.default_root_dir "${LOG_DIR}/${PRE_DATA}" \
    #         --data.data_manager.init_args.path ${DATA_DIR} \
    #         --data.data_manager.init_args.split_val true \
    #         --data.data_manager.init_args.context_length ${CTX_LEN} \
    #         --data.data_manager.init_args.prediction_length ${PRED_LEN} \
    #         --data.data_manager.init_args.dataset=${DATASET} \
    #         --model.forecaster.init_args.ckpt_path "/data/Blob_EastUS/v-zhenwzhang/log/moirai_pretrain/moirai_small/${PRE_DATA}/${PRE_DATA}/checkpoints/epoch=999-step=100000.ckpt" \
    #         --data.test_batch_size 1
    # done
    # for PRE_DATA in others # cmip6 era5
    # do
    #     python run.py --config config/tsfm/moirai.yaml \
    #         --seed_everything 0 \
    #         --trainer.default_root_dir "${LOG_DIR}/${PRE_DATA}" \
    #         --data.data_manager.init_args.path ${DATA_DIR} \
    #         --data.data_manager.init_args.split_val true \
    #         --data.data_manager.init_args.context_length ${CTX_LEN} \
    #         --data.data_manager.init_args.prediction_length ${PRED_LEN} \
    #         --data.data_manager.init_args.dataset=${DATASET} \
    #         --model.forecaster.init_args.ckpt_path "/data/Blob_EastUS/v-zhenwzhang/log/moirai_pretrain/moirai_small/${PRE_DATA}/${PRE_DATA}/checkpoints/epoch=499-step=50000.ckpt" \
    #         --data.test_batch_size 1
    # done
    # for PRE_DATA in lib_city # cmip6 era5
    # do
    #     python run.py --config config/tsfm/moirai.yaml \
    #         --seed_everything 0 \
    #         --trainer.default_root_dir "${LOG_DIR}/${PRE_DATA}" \
    #         --data.data_manager.init_args.path ${DATA_DIR} \
    #         --data.data_manager.init_args.split_val true \
    #         --data.data_manager.init_args.context_length ${CTX_LEN} \
    #         --data.data_manager.init_args.prediction_length ${PRED_LEN} \
    #         --data.data_manager.init_args.dataset=${DATASET} \
    #         --model.forecaster.init_args.ckpt_path "/data/Blob_EastUS/v-zhenwzhang/log/moirai_pretrain/moirai_small/${PRE_DATA}/${PRE_DATA}/checkpoints/epoch=899-step=90000.ckpt" \
    #         --data.test_batch_size 1
    # done
    python run.py --config config/tsfm/moirai.yaml \
        --seed_everything 0 \
        --trainer.default_root_dir "${LOG_DIR}/small_ori" \
        --data.data_manager.init_args.path ${DATA_DIR} \
        --data.data_manager.init_args.split_val true \
        --data.data_manager.init_args.context_length ${CTX_LEN} \
        --data.data_manager.init_args.prediction_length ${PRED_LEN} \
        --data.data_manager.init_args.dataset=${DATASET} \
        --model.forecaster.init_args.model_size small \
        --data.test_batch_size 1
done

#  etth1 etth2 

# python run.py --config config/pretrain/moirai.yaml \
#         --seed_everything 0