#!/bin/bash

source bin/utils.sh

task="task-6"
mkdir -p checkpoints/$task/
WITH_PREPROCESS="$1"
[ -z "$WITH_PREPROCESS" ] && WITH_PREPROCESS='False'

if [ "$WITH_PREPROCESS" == "True" ]; then
  python parse_candidates.py data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-candidates.txt > data/candidates-dstc2.tsv
  parse_dialogs 'dialog-babi-task6-dstc2' $task "--ignore_options"
fi

python train.py --train data/train-$task.tsv --dev data/dev-$task-500.tsv \
  --vocab data/vocab-$task.tsv --emb_dim 128 --save_dir checkpoints/$task/model \
  --margin 0.01 --negative_cand 100 --learning_rate 0.001 --candidates data/candidates-dstc2.tsv
