export CUDA_VISIBLE_DEVICES=0

DATASET_ft=${1}
if [ -n "${2}" ]; then
    SEED="_seed_${2}"
else
    SEED=""
fi

DATA_DIR=/data/Blob_EastUS/v-jiawezhang/data/all_datasets/
CTX_LEN=96
PRED_LEN=96
MOIRAI_MODEL=moirai_1.1_R_base

LOG_DIR=/data/Blob_EastUS/v-zhenwzhang/log/probts_test/moirai_finetune/group5v1_fix

file_path="/data/Blob_EastUS/v-zhenwzhang/log/moirai_finetune/${MOIRAI_MODEL}/${DATASET_ft}/${MOIRAI_MODEL}_ft_${DATASET_ft}_lr_1em4_fix${SEED}/checkpoints"
ckpt_file="${file_path}/epoch=9-step=1000.ckpt" # /epoch=99-step=10000.ckpt /epoch=39-step=4000.ckpt
for DATASET_eval in etth1 etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
do
    for CTX_LEN in 96 512 1000 2000
    do
        for variate_mode in M S
        do
            for PRED_LEN in 24 48 96 192 336 720
            do
                python run.py --config config/tsfm/moirai.yaml \
                    --seed_everything 0 \
                    --trainer.default_root_dir "${LOG_DIR}/${MOIRAI_MODEL}_ft_${DATASET_ft}_fix_ep10${SEED}_CTX_${CTX_LEN}_VAR_${variate_mode}_PRED_${PRED_LEN}" \
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
done

file_path="/data/Blob_EastUS/v-zhenwzhang/log/moirai_finetune/${MOIRAI_MODEL}/${DATASET_ft}/${MOIRAI_MODEL}_ft_${DATASET_ft}_lr_1em4_fix${SEED}/checkpoints"
ckpt_file="${file_path}/epoch=19-step=2000.ckpt" # /epoch=99-step=10000.ckpt /epoch=39-step=4000.ckpt
for DATASET_eval in etth1 etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
do
    for CTX_LEN in 96 512 1000 2000
    do
        for variate_mode in M S
        do
            for PRED_LEN in 24 48 96 192 336 720
            do
                python run.py --config config/tsfm/moirai.yaml \
                    --seed_everything 0 \
                    --trainer.default_root_dir "${LOG_DIR}/${MOIRAI_MODEL}_ft_${DATASET_ft}_fix_ep20${SEED}_CTX_${CTX_LEN}_VAR_${variate_mode}_PRED_${PRED_LEN}" \
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
done

file_path="/data/Blob_EastUS/v-zhenwzhang/log/moirai_finetune/${MOIRAI_MODEL}/${DATASET_ft}/${MOIRAI_MODEL}_ft_${DATASET_ft}_lr_1em4_fix${SEED}/checkpoints"
ckpt_file="${file_path}/epoch=99-step=10000.ckpt" # /epoch=99-step=10000.ckpt /epoch=39-step=4000.ckpt
for DATASET_eval in etth1 etth2 ettm1 ettm2 traffic_ltsf electricity_ltsf exchange_ltsf weather_ltsf
do
    for CTX_LEN in 96 512 1000 2000
    do
        for variate_mode in M S
        do
            for PRED_LEN in 24 48 96 192 336 720
            do
                python run.py --config config/tsfm/moirai.yaml \
                    --seed_everything 0 \
                    --trainer.default_root_dir "${LOG_DIR}/${MOIRAI_MODEL}_ft_${DATASET_ft}_fix_ep100${SEED}_CTX_${CTX_LEN}_VAR_${variate_mode}_PRED_${PRED_LEN}" \
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
done

# IMPORTANT: check this config!!!!
# --model.forecaster.init_args.ckpt_path ${ckpt_file} \
# --model.forecaster.init_args.model_size small \
