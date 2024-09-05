export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps/moirai_ori_pretrain_progress
CTX_LEN=96
PRED_LEN=96

for PRE_DATA in buildings_bench buildings_900k lib_city gluonts cloudops_tsf largest others proenfo subseasonal # cmip6 era5
do
    for epoch in 99 199 299 399 499 599 699 799 899 999
    do
        path="/data/Blob_EastUS/v-zhenwzhang/log/moirai_pretrain/moirai_small/${PRE_DATA}/${PRE_DATA}/checkpoints/"
        PATTERN=".*epoch=${epoch}.*v1\.ckpt.*"
        file=$(find "${path}" -maxdepth 1 -type f -name "*.ckpt" | grep -E ${PATTERN})
        if [ -z "$file" ]; then
            PATTERN=".*epoch=${epoch}.*\.ckpt.*"
            file=$(find "$path" -maxdepth 1 -type f -name "*.ckpt" | grep -E ${PATTERN})
        fi
        if [ -n "$file" ]; then
            echo "Found file: $file"
        for DATASET in etth1 etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
        do
            python run.py --config config/tsfm/moirai.yaml \
            --seed_everything 0 \
            --trainer.default_root_dir "${LOG_DIR}/${PRE_DATA}_${epoch}" \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --data.data_manager.init_args.split_val true \
            --data.data_manager.init_args.context_length ${CTX_LEN} \
            --data.data_manager.init_args.prediction_length ${PRED_LEN} \
            --data.data_manager.init_args.dataset=${DATASET} \
            --model.forecaster.init_args.ckpt_path ${file} \
            --data.test_batch_size 1
        done
        else
            echo "No file found for ${PRE_DATA} epoch=${epoch}"
        fi
    done
done

#  etth1 etth2 

# python run.py --config config/pretrain/moirai.yaml \
#         --seed_everything 0