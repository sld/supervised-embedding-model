# Description

It is the implementation of Supervised embedding models from
[[Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/abs/1605.07683v3)] paper.

Results almost the same as in the paper.

Here you can find Russian paper-note of the paper: [link](https://github.com/sld/deeplearning-papernotes/blob/master/notes/end-to-end-goal.md).

# Environment

* Python 3.6.0
* tensorflow 1.0.0
* Dialog bAbI Tasks Data 1-6 corpus, download by the [link](https://research.fb.com/downloads/babi/).
This corpus should be placed in data/dialog-bAbI-tasks directory.


All packages are listed in requirements.txt.

# Reproduce results

0. Setup the environment.
1. Run: `bin/train_all.sh`
2. After approx. 1 hour run it in test set: `bin/test_all.sh`


