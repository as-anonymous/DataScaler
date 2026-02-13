#!/bin/bash

[ -d "poc-datascaler-proxy" ] || { echo "Error: poc-datascaler-proxy directory not found"; exit 1; }

cd poc-datascaler-proxy || exit 1

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

declare -a lrs
declare -a seeds
nproc_per_node=""
scale=""
data_dir=""
output_dir=""
seed=""
width_mult=""
min_width_mult=""
train_num_samples=""
num_checkpoints=""
save_frequency=""
workers=""
attn_mult=1.0
output_mult=1.0
use_mup=true
log_every_n_steps=1
val_ratio=0.0
val_batch_size=16384
lr_scheduler=cosine
subset_file_path=None
subset_file_name=None
min_fraction=0.0
fraction=1.0
method=clip
model_arch=ViT-B-32

for arg in "$@"; do
    case $arg in
        --lrs=*)        IFS=' ' read -r -a lrs <<< "${arg#*=}" ;;
        --seeds=*)        IFS=' ' read -r -a seeds <<< "${arg#*=}" ;;
        --nproc-per-node=*) nproc_per_node="${arg#*=}" ;;
        --scale=*) scale="${arg#*=}" ;;
        --data-dir=*) data_dir="${arg#*=}" ;;
        --output-dir=*) output_dir="${arg#*=}" ;;
        --width-mult=*) width_mult="${arg#*=}" ;;
        --min-width-mult=*) min_width_mult="${arg#*=}" ;;
        --train-num-samples=*) train_num_samples="${arg#*=}" ;;
        --num-checkpoints=*) num_checkpoints="${arg#*=}" ;;
        --save-frequency=*) save_frequency="${arg#*=}" ;;
        --workers=*) workers="${arg#*=}" ;;
        --attn-mult=*) attn_mult="${arg#*=}" ;;
        --output-mult=*) output_mult="${arg#*=}" ;;
        --log-every-n-steps=*) log_every_n_steps="${arg#*=}" ;;
        --val-ratio=*) val_ratio="${arg#*=}" ;;
        --val-batch-size=*) val_batch_size="${arg#*=}" ;;
        --lr-scheduler=*) lr_scheduler="${arg#*=}" ;;
        --subset-file-path=*) subset_file_path="${arg#*=}" ;;
        --subset-file-name=*) subset_file_name="${arg#*=}" ;;
        --min-fraction=*) min_fraction="${arg#*=}" ;;
        --fraction=*) fraction="${arg#*=}" ;;
        --method=*) method="${arg#*=}" ;;
        --model-arch=*) model_arch="${arg#*=}" ;;
        --use-mup=true) use_mup=true ;;
        --use-mup=false) use_mup=false ;;
        *) echo "❌  Unknown option: $arg" && exit 1 ;;
    esac
done

for lr in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
        if $use_mup = "true"; then
            torchrun --nproc_per_node "$nproc_per_node" --no-python datascaler-datacomp train --scale "$scale" --data-dir "$data_dir" --output-dir "$output_dir" --exp-name "${scale}_${subset_file_name}_${seed}" \
            --seed "$seed" --width-mult "$width_mult" --min-width-mult "$min_width_mult" --train-num-samples "$train_num_samples" --num-checkpoints "$num_checkpoints" --save-frequency "$save_frequency" --learning-rate "$lr" --workers "$workers" --model-arch "$model_arch" \
            --attn-mult "$attn_mult" --output-mult "$output_mult" --log-every-n-steps "$log_every_n_steps" --val-ratio "$val_ratio" --val-batch-size "$val_batch_size" --lr-scheduler "$lr_scheduler" --subset-file "${subset_file_path}/${subset_file_name}_trie.pkl" --use-mup
        else
            torchrun --nproc_per_node "$nproc_per_node" --no-python datascaler-datacomp train --scale "$scale" --data-dir "$data_dir" --output-dir "$output_dir" --exp-name "${scale}_${subset_file_name}_${seed}" \
            --seed "$seed" --width-mult "$width_mult" --min-width-mult "$min_width_mult" --train-num-samples "$train_num_samples" --num-checkpoints "$num_checkpoints" --save-frequency "$save_frequency" --learning-rate "$lr" --workers "$workers" --model-arch "$model_arch" \
            --attn-mult "$attn_mult" --output-mult "$output_mult" --log-every-n-steps "$log_every_n_steps" --val-ratio "$val_ratio" --val-batch-size "$val_batch_size" --lr-scheduler "$lr_scheduler" --subset-file "${subset_file_path}/${subset_file_name}_trie.pkl" --no-use-mup
        fi
    done
done
