#!/bin/bash

mkdir -p checkpoints/task-2/
WITH_PREPROCESS="$1"
[ -z "$WITH_PREPROCESS" ] && WITH_PREPROCESS='False'

if [ "$WITH_PREPROCESS" == "True" ]; then
	python parse_candidates.py data/dialog-bAbI-tasks/dialog-babi-candidates.txt > data/candidates.tsv
	python parse_dialogs.py data/dialog-bAbI-tasks/dialog-babi-task2-API-refine-trn.txt False > data/train-task-2.tsv
	python parse_dialogs.py data/dialog-bAbI-tasks/dialog-babi-task2-API-refine-dev.txt False > data/dev-task-2.tsv
	python parse_dialogs.py data/dialog-bAbI-tasks/dialog-babi-task2-API-refine-tst.txt False > data/test-task-2.tsv
	shuf -n 500 data/dev-task-2.tsv > data/dev-task2-500.tsv
	cat data/train-task-2.tsv data/dev-task-2.tsv data/test-task-2.tsv | python build_vocabulary.py > data/vocab-task2.tsv
fi

python train.py --train data/train-task-2.tsv --dev data/dev-task2-500.tsv --vocab data/vocab-task2.tsv --emb_dim 128 --save_dir checkpoints/task-2/model