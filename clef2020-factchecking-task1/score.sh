#!/bin/bash
GOLD_DIR="./test-input/"
PRED_DIR="../saved_models/clef20_dune_weighted_sampler/preds/"

python3 scorer/main.py --gold_file_path="${GOLD_DIR}test-gold.tsv" --pred_file_path="${PRED_DIR}ranking.tsv"
