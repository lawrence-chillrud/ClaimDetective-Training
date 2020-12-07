#!/bin/bash

GOLD_DIR="./data/test_annotated/"
PRED_DIR="../saved_models/clef20_clef19_dune_weighted_sampler/preds/"

python3 scorer/main.py --gold_file_path="${GOLD_DIR}20151219_3_dem.tsv,
                                         ${GOLD_DIR}20160129_7_gop.tsv,
                                         ${GOLD_DIR}20160311_12_gop.tsv,
                                         ${GOLD_DIR}20180131_state_union.tsv,
                                         ${GOLD_DIR}20181015_60_min.tsv,
                                         ${GOLD_DIR}20190205_trump_state.tsv,
                                         ${GOLD_DIR}20190215_trump_emergency.tsv" \
                       --pred_file_path="${PRED_DIR}20151219_3_dem_out.tsv,
                                         ${PRED_DIR}20160129_7_gop_out.tsv,
                                         ${PRED_DIR}20160311_12_gop_out.tsv,
                                         ${PRED_DIR}20180131_state_union_out.tsv,
                                         ${PRED_DIR}20181015_60_min_out.tsv,
                                         ${PRED_DIR}20190205_trump_state_out.tsv,
                                         ${PRED_DIR}20190215_trump_emergency_out.tsv"
