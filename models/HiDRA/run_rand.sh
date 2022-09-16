#!/bin/bash

start_time=$SECONDS

echo "running random pathway baselines......"

python main_rand.py --dataroot ../../input_data/ \
--outroot results/fp/rand_pathway_baseline/ \
--hyproot best_hyp_fp/PID/model_hyp_LDO.txt \
--pathway PID \
--foldtype drug \
--drug_feature_type fp \


echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
