export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps

for DATASET in 'solar' 'electricity' 'exchange' 'traffic' 'wiki'
do
    for MODEL in 'dlinear' 'patchtst' 'gru_nvp' 'gru_maf' 'trans_maf' 'timegrad' 'csdi' 'timesnet'
    do
        python run.py --config config/stsf/${DATASET}/${MODEL}.yaml --seed_everything 0  \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --trainer.default_root_dir ${LOG_DIR} \
            --data.data_manager.init_args.split_val true 
    done
done
