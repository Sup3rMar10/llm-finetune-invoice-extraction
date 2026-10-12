#!/usr/bin/env bash
# Reproduce the full experiment: generate data -> LoRA train -> evaluate 3 configs on 2 splits -> report.
# Defaults are sized for a laptop CPU (~2 h). On a GPU, drop --max-train and raise --epochs.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="outputs/qwen0.5b-lora"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8

$PY -m src.generate_data

$PY -m src.train --model "$MODEL" --out "$ADAPTER" --max-train "${MAX_TRAIN:-200}" --epochs "${EPOCHS:-1}"
cp "$ADAPTER/train_log.jsonl" results/train_log.jsonl
cp "$ADAPTER/train_config.json" results/train_config.json

for split in test_id test_ood; do
  $PY -m src.evaluate --model "$MODEL" --name tuned         --split "data/$split.jsonl" --adapter "$ADAPTER"
  $PY -m src.evaluate --model "$MODEL" --name base-zeroshot --split "data/$split.jsonl"
  $PY -m src.evaluate --model "$MODEL" --name base-fewshot  --split "data/$split.jsonl" --fewshot 2
done

$PY -m src.report
echo "DONE"
