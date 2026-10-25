export CUDA_VISIBLE_DEVICES=6
EXP=3t3

python /group/dingyi/ct/src/train.py with dataset="${EXP}" >"train_${EXP}.log" 2>&1
