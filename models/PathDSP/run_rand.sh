#!/bin/bash

start_time=$SECONDS

echo "running random pathway baselines......"

python main_rand.py --dataroot ../../input_data/ \
--outroot results/PID_rand/LDO/ \
--hyproot best_hyp/PID/model_hyp_LDO.txt \
--pathway PID \
--foldtype drug \


echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
