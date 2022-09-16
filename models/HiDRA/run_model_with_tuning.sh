#!/bin/bash

start_time=$SECONDS

echo "running model from scratch with hyperparameter tuning......"
echo "best hyperparameter will be selected and applied to the model......"

python main.py --dataroot ../../input_data/ \
--outroot results/tuning/fp/ \
--pathway KEGG \
--foldtype pair \
--drug_feature_type fp \
--tuning \
--gridroot hyp_grid/fp_KEGG_grid.txt \
--max_tuning_epoch 1 \
--num_samples 1 \


echo "finished run"
elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"