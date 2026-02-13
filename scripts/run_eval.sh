#!/bin/bash

declare -a lrs
declare -a seeds
width_mult=""
min_width_mult=""
train_dir=""
task_dir=""
data_dir=""
min_epoch=1
max_epoch=""
num_workers=8
scale=""
use_mup=true
attn_mult=1.0
output_mult=1.0
max_parallel=5
compute_loss=0.0
min_fraction=0.0
fraction=1.0
method=clip
model_arch=ViT-B-32

for arg in "$@"; do
    case $arg in
        --lrs=*)        IFS=' ' read -r -a lrs <<< "${arg#*=}" ;;
        --seeds=*)        IFS=' ' read -r -a seeds <<< "${arg#*=}" ;;
        --width-mult=*) width_mult="${arg#*=}" ;;
        --min-width-mult=*) min_width_mult="${arg#*=}" ;;
        --train-dir=*) train_dir="${arg#*=}" ;;
        --task-dir=*) task_dir="${arg#*=}" ;;
        --data-dir=*) data_dir="${arg#*=}" ;;
        --min-epoch=*) min_epoch="${arg#*=}" ;;
        --max-epoch=*) max_epoch="${arg#*=}" ;;
        --num-workers=*) num_workers="${arg#*=}" ;;
        --scale=*) scale="${arg#*=}" ;;
        --attn-mult=*) attn_mult="${arg#*=}" ;;
        --output-mult=*) output_mult="${arg#*=}" ;;
        --compute-loss=*) compute_loss="${arg#*=}" ;;
        --min-fraction=*) min_fraction="${arg#*=}" ;;
        --fraction=*) fraction="${arg#*=}" ;;
        --method=*) method="${arg#*=}" ;;
        --model-arch=*) model_arch="${arg#*=}" ;;
        --use-mup=true) use_mup=true ;;
        --use-mup=false) use_mup=false ;;
        --max-parallel=*) max_parallel="${arg#*=}" ;;
        *) echo "❌   Unknown option: $arg" && exit 1 ;;
    esac
done

[ -d "poc-datascaler-proxy" ] || { echo "Error: poc-datascaler-proxy directory not found"; exit 1; }

cd poc-datascaler-proxy || exit 1

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

for lr in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
        python scripts/eval.py "${train_dir}/${scale}_${width_mult}_${min_width_mult}_${lr}_${attn_mult}_${output_mult}_${use_mup}_${method}_${min_fraction}_${fraction}_${model_arch}_${seed}" \
        "$task_dir" "$data_dir" "$min_epoch" "$max_epoch" "$width_mult" "$num_workers" "$scale" "$use_mup" "$attn_mult" "$output_mult" "$max_parallel" "$compute_loss" "$model_arch"
    done
done
