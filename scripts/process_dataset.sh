#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null && pwd -P)"
DIR="$(cd "$SCRIPT_DIR/.." > /dev/null && pwd -P)"
cd "$DIR"

BASE_DIR=$DIR/data/RitmeTombak2/Loop/3.4
SPLIT_DIR=$DIR/split_files/RitmeTombak2/Loop/3.4
TVER="W3_TH1.5_FFT4"

# python scripts/calc_vad.py -i "data/RitmeTombak2/Loop/3.4"
# python scripts/transient_detector.py -i $BASE_DIR -l $TVER \
#                                         --delta 0.001 \
#                                         --sigma 0.05

# python scripts/merge_transient_roll.py -a $BASE_DIR/audio
#                                         -t $BASE_DIR/seq/lab_phon \
#                                         -r $BASE_DIR/seq/lab_roll/manual \
#                                         -o $BASE_DIR/seq/lab_seg

rm -r $BASE_DIR/seq/lab_phon/*
rm -r $SPLIT_DIR/*

python scripts/split_transients.py -i $BASE_DIR \
                                    -o $SPLIT_DIR \
                                    -t $TVER

                                    
python scripts/cluster_segments.py -i $SPLIT_DIR \
                                    -o $BASE_DIR \
                                    -c 2 \
                                    -a "gmm"

python scripts/create_roll_sections.py -i $BASE_DIR \
                                        -c 1