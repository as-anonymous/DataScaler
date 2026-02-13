#!/bin/bash

data_dir=""
output_dir=""
scale=""
num_workers=""
batch_size=""
method=""
min_fraction=0.0
fraction=""

[ -d "poc-datascaler-proxy" ] || { echo "Error: poc-datascaler-proxy directory not found"; exit 1; }

cd poc-datascaler-proxy || exit 1

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

for arg in "$@"; do
    case $arg in
        --data-dir=*) data_dir="${arg#*=}" ;;
        --output-dir=*) output_dir="${arg#*=}" ;;
        --scale=*) scale="${arg#*=}" ;;
        --num-workers=*) num_workers="${arg#*=}" ;;
        --batch-size=*) batch_size="${arg#*=}" ;;
        --method=*) method="${arg#*=}" ;;
        --min-fraction=*) min_fraction="${arg#*=}" ;;
        --fraction=*) fraction="${arg#*=}" ;;
        *) echo "❌   Unknown option: $arg" && exit 1 ;;
    esac
done

python -m datascaler_open_clip.generate_pools --data-dir ${data_dir} --output-dir ${output_dir} --num-workers ${num_workers} --data-scale ${scale} --batch-size ${batch_size} --method ${method} --min-fraction ${min_fraction} --fraction ${fraction}

echo "Generation done"
