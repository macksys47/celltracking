export CUDA_VISIBLE_DEVICES=1
EXP=3t3

python /group/dingyi/ct/src/pipeline.py with dataset="${EXP}" >"inf_${EXP}.log" 2>&1
