#!/bin/bash
cd "$(dirname "$0")/.."
python src/inference.py --config configs/baseline.yaml "$@"
