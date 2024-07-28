export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps

python run.py --config config/pretrain/patchtst.yaml --seed_everything 0  \
