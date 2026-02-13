# PoC datascaler Proxy

## Prerequisite
Download DataComp datsaets (train and eval) from https://github.com/mlfoundations/datacomp.

## Installation

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync


# [Error Case] 'invalid peer certificate: UnknownIssuer'
# option 1, bypass certificate check
uv sync --allow-insecure-host pypi.org

# option 2, use system certificate
uv sync --native-tls

# option 3. configure --native-tls to be always true
export UV_NATIVE_TLS=true
uv sync

# option 4, explicitly set certificate path
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
uv sync


# use virtual environment
. .venv/bin/activate

# install pre-commit hooks
pre-commit install
```

## Compute data scores
`generate_pools.sh` computes data scores for `--method`.
```bash
bash poc-datascaler-proxy/scripts/generate_pools.sh --data-dir=<DATA_DIR> \
                                                    --output-dir=<OUTPUT_DIR> \
                                                    --scale=medium \
                                                    --method=dfn \
                                                    --fraction=1.0

# --data-dir: Path to tar files containing shards
# --output-dir: Path to score and subset files
# --method: Method to compute scores (CLIP, TMARS, DFN)
# --scale: Dataset scale (small, medium)
# --fraction: Fraction of data to keep
# (if less than 1.0, it generates a subset file (`.pkl`) containing unique ids, top 'fraction' samples based on `method` scores)
```

## Train using TRIE files (No resharding required)
To prevent resharding the data using the generated filters. We have adapted openclip dataloader to use a dictionary passed to the training script using the subset_file argument to determine whether a sample should be included in training.

To pretrain clip model on datacomp-medium using the generated subset file using multiple node run:
```bash

bash poc-datascaler-proxy/scripts/train_with_grid_search.sh --nproc-per-node=<NUM_NODES> \
                                                            --data-dir=<DATA_DIR> \
                                                            --output-dir=<OUTPUT_DIR> \
                                                            --width-mult=1.0
                                                            --min-width-mult=0.1875 \
                                                            --scale=medium \
                                                            --train-num-samples=128_000_000 \
                                                            --use-mup=true \
                                                            --subset-file-path=<SUBSET_FILE> \
                                                            --method=clip \
                                                            --fraction=0.4 \
                                                            --model-arch "ViT-B-32"

# --data-dir: Path to data directory containing shards
# --output-dir: Output directory for saving models and logs
# --width-mult: Width multiplier for generating CLIP model
# --min-width-mult: Minimum width multiplier for generating CLIP model
# --scale: Dataset scale (small, medium)
# --train-num-samples: Compute budget (i.e., the number of samples seen during training)
# --use_mup: use mup
# --subset-file: Path to the generated subset file (default: None)
# --method: Method to compute scores (CLIP, TMARS, DFN)
# --fraction: Fraction of data to keep
# --model-arch: Architecture of the trained CLIP model (default: ViT-B-32)
```

## Evaluate trained model using Datacomp evaluation datasets
```bash
bash poc-datascaler-proxy/scripts/run_eval.sh --train-dir=<TRAIN_OUTPUT_DIR> \
                                              --task-dir=<TASK_DIR> \
                                              --data-dir <EVAL_DATA_DIR> \
                                              --width-mult=1.0 \
                                              --min-width-mult=0.1875 \
                                              --use-mup=true \
                                              --scale=medium \
                                              --fraction=0.4 \
                                              --method=clip \
                                              --model-arch "ViT-B-32"

# --train-dir: Path to output directory from training / checkpoints
# --task-dir: Path to directory containing evaluation tasks
# --data-dir: Path to directory containing downloaded evaluation datasets
# --width-mult: Width multiplier for generating CLIP model
# --min-width-mult: Minimum width multiplier for generating CLIP model
# --use_mup: use mup
# --scale: Dataset scale (small, medium)
# --fraction: Fraction of data to keep
# --method: Method to compute scores (CLIP, TMARS, DFN)
# --model-arch: Architecture of the trained CLIP model (default: ViT-B-32)```

