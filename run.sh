#!/bin/bash
# Script to run in-context learning experiments with G-Align

# Set the path to your pretrained model
MODEL_PATH="generated_files/output/G-Align/final_gfm_model.pt"

# Test on Cora dataset with 5-shot learning
echo "Testing on Cora with 5-shot learning..."
python icl.py \
    --model_path $MODEL_PATH \
    --dataset cora \
    --k_shot 5 \
    --n_runs 10 \
    --device cuda \
    --seed 42

# Test with different k values
echo -e "\nTesting with different k-shot values..."
for k in 1 3 5 10; do
    echo -e "\n${k}-shot learning:"
    python icl.py \
        --model_path $MODEL_PATH \
        --dataset cora \
        --k_shot $k \
        --n_runs 5 \
        --device cuda \
        --seed 42
done

# Test on other datasets if available
# echo -e "\nTesting on CiteSeer..."
# python icl.py \
#     --model_path $MODEL_PATH \
#     --dataset citeseer \
#     --k_shot 5 \
#     --n_runs 10 \
#     --device cuda \
#     --seed 42
