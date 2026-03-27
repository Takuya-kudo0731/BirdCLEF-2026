#!/bin/bash
cd "$(dirname "$0")/.."
python src/train.py --config configs/baseline.yaml "$@"
