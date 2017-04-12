#!/bin/bash

mkdir -p checkpoints/task-1/
WITH_PREPROCESS="$1"
[ -z "$WITH_PREPROCESS" ] && WITH_PREPROCESS='False'

if [ "$WITH_PREPROCESS" == "True" ]; then
	python parse_candidates.py data/dialog-bAbI-tasks/dialog-babi-candidates.txt > data/candidates.tsv
	python parse_dialogs.py data/dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt True > data/train-task-1-history.tsv
	python parse_dialogs.py data/dialog-bAbI-tasks/dialog-babi-task1-API-calls-dev.txt True > data/dev-task-1-history.tsv
	python parse_dialogs.py data/dialog-bAbI-tasks/dialog-babi-task1-API-calls-tst.txt True > data/test-task-1-history.tsv
	shuf -n 500 data/dev-task-1-history.tsv > data/dev-task1-history-500.tsv
	cat data/train-task-1-history.tsv data/dev-task-1-history.tsv data/test-task-1-history.tsv | python build_vocabulary.py > data/vocab-task1.tsv
fi

python train.py --train data/train-task-1-history.tsv --dev data/dev-task1-history-500.tsv --vocab data/vocab-task1.tsv --save_dir checkpoints/task-1/model