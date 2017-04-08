from make_train_tensor import make_tensor, load_vocab
from model import Model
from sys import argv
from utils import batch_iter
from tqdm import tqdm
import numpy as np
import tensorflow as tf


def main(test_tensor, candidates_tensor, model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(evaluate(test_tensor, candidates_tensor, sess, model))


def evaluate(test_tensor, candidates_tensor, sess, model):
    neg = 0
    pos = 0
    for row in tqdm(test_tensor):
        true_context = [row[0]]
        test_score = sess.run(
            model.f,
            feed_dict={model.context_batch: true_context, model.response_batch: [row[1]]}
        )
        test_score = test_score[0]

        is_pos = evaluate_one_row(candidates_tensor, true_context, sess, model, test_score)
        if is_pos:
            pos += 1
        else:
            neg += 1
    return (pos, neg, pos / (pos+neg))


def evaluate_one_row(candidates_tensor, true_context, sess, model, test_score):
    for batch in batch_iter(candidates_tensor, 64):
        candidate_responses = batch[:, 0, :]
        context_batch = np.repeat(true_context, candidate_responses.shape[0], axis=0)

        scores = sess.run(
            model.f,
            feed_dict={model.context_batch: context_batch, model.response_batch: candidate_responses}
        )
        for score in scores:
            if score > test_score:
                return False
    return True


if __name__ == '__main__':
    test_filename = argv[1]
    candidates_filename = argv[2]
    vocab_filename = argv[3]
    vocab = load_vocab(vocab_filename)
    test_tensor = make_tensor(test_filename, vocab_filename)
    candidates_tensor = make_tensor(candidates_filename, vocab_filename)
    model = Model(len(vocab), 32)
    main(test_tensor, candidates_tensor, model)
