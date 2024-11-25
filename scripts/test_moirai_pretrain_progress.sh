export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps/moirai_ori_pretrain_progress
CTX_LEN=96
PRED_LEN=96

# for PRE_DATA in lotsa_v1_varying_weights_1_constant_lr lotsa_v1_varying_weights_1_constant_lr_1em4  # others buildings_900k lib_city gluonts cloudops_tsf largest others proenfo subseasonal # cmip6 era5
# do
#     for epoch in 99 199 299 399 499 599 699 799 899 999 # 99 199 299 399 499 599 699 799 899 999
#     do
#         path="/data/Blob_EastUS/v-zhenwzhang/log/moirai_pretrain/moirai_small/lotsa_v1_weighted/${PRE_DATA}/checkpoints/"
#         # path="/data/Blob_EastUS/v-zhenwzhang/log/moirai_finetune/moirai_1.0_R_small/etth1/etth1_val_etth1/checkpoints"
#         PATTERN=".*epoch=${epoch}.*v1\.ckpt.*"
#         file=$(find "${path}" -maxdepth 1 -type f -name "*.ckpt" | grep -E "${PATTERN}" | head -n 1)
#         if [ -z "$file" ]; then
#             PATTERN=".*epoch=${epoch}.*\.ckpt.*"
#             file=$(find "$path" -maxdepth 1 -type f -name "*.ckpt" | grep -E "${PATTERN}" | head -n 1)
#         fi
#         if [ -n "$file" ]; then
#             echo "Found file: $file"
#             for DATASET in etth1 etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
#             do
#                 echo "${file} Testing ${PRE_DATA} epoch=${epoch} on ${DATASET}..."
#                 python run.py --config config/tsfm/moirai.yaml \
#                 --seed_everything 0 \
#                 --trainer.default_root_dir "${LOG_DIR}/${PRE_DATA}_${epoch}" \
#                 --data.data_manager.init_args.path ${DATA_DIR} \
#                 --data.data_manager.init_args.split_val true \
#                 --data.data_manager.init_args.context_length ${CTX_LEN} \
#                 --data.data_manager.init_args.prediction_length ${PRED_LEN} \
#                 --data.data_manager.init_args.dataset=${DATASET} \
#                 --model.forecaster.init_args.ckpt_path "$file" \
#                 --data.test_batch_size 1
#             done
#         else
#             echo "No file found for ${PRE_DATA} epoch=${epoch}"
#         fi
#     done
# done

LOG_DIR=./exps/moirai_ori_finetune

# for DATASET_ft in etth1 etth2 ettm1 ettm2 traffic electricity exchange weather # 
for DATASET_ft in lotsa_v1_weighted_group1 lotsa_v1_weighted_group2 lotsa_v1_weighted_group3 lotsa_v1_weighted_group4 lotsa_v1_weighted_group5
do
    file_path="/data/Blob_EastUS/v-zhenwzhang/log/moirai_finetune/moirai_1.0_R_small/${DATASET_ft}/moirai_1.0_R_small_ft_${DATASET_ft}/checkpoints"
    ckpt_file="${file_path}/epoch=49-step=5000.ckpt"
    # ckpt_file=$(find "$file_path" -maxdepth 1 -type f -name "*.ckpt" | head -n 1)
    # if [ -z "$ckpt_file" ]; then
    #     echo "No .ckpt file found in $file_path"
    # else
    #     echo "Found .ckpt file: $ckpt_file"
        for DATASET_eval in etth1 etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
        do
            for CTX_LEN in 96 336 512
            do
                for variate_mode in M S
                do 
                    python run.py --config config/tsfm/moirai.yaml \
                        --seed_everything 0 \
                        --trainer.default_root_dir "${LOG_DIR}/moirai_1.0_R_small_ft_${DATASET_ft}_CTX_${CTX_LEN}_VAR_${variate_mode}" \
                        --data.data_manager.init_args.path ${DATA_DIR} \
                        --data.data_manager.init_args.split_val true \
                        --data.data_manager.init_args.context_length ${CTX_LEN} \
                        --data.data_manager.init_args.prediction_length ${PRED_LEN} \
                        --data.data_manager.init_args.dataset ${DATASET_eval} \
                        --model.forecaster.init_args.ckpt_path ${ckpt_file} \
                        --model.forecaster.init_args.variate_mode ${variate_mode} \
                        --data.test_batch_size 1
                done
            done
        done
    # fi
done

# file_path="/data/Blob_EastUS/v-zhenwzhang/log/moirai_finetune/moirai_1.0_R_small/etth1/moirai_1.0_R_small_ft_etth1_val_etth1_multi/checkpoints/epoch=19-step=2000.ckpt"
# for DATASET in etth1 etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
# do
#     python run.py --config config/tsfm/moirai.yaml \
#     --seed_everything 0 \
#     --trainer.default_root_dir "${LOG_DIR}/moirai_1.0_R_large" \
#     --data.data_manager.init_args.path ${DATA_DIR} \
#     --data.data_manager.init_args.split_val true \
#     --data.data_manager.init_args.context_length ${CTX_LEN} \
#     --data.data_manager.init_args.prediction_length ${PRED_LEN} \
#     --data.data_manager.init_args.dataset ${DATASET} \
#     --model.forecaster.init_args.model_size large \
#     --data.test_batch_size 1
# done