from make_tensor import make_tensor, load_vocab
from model import Model
from sys import argv
from utils import batch_iter
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import argparse
import sys


def main(test_tensor, candidates_tensor, model, checkpoint_dir, dev_topic_tensor):
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        saver.restore(sess, ckpt.model_checkpoint_path)
        dev_eval = evaluate(test_tensor, candidates_tensor, sess, model, dev_topic_tensor)
        print("Eval: {}".format(dev_eval))


def evaluate(test_tensor, candidates_tensor, sess, model, dev_topic_tensor):
    neg = 0
    pos = 0
    ind = 0
    for row in tqdm(test_tensor):
        true_context = [row[0]]
        true_topic = [dev_topic_tensor[ind, 0, :]]
        test_score = sess.run(
            model.f_pos,
            feed_dict={model.context_batch: true_context,
                       model.response_batch: [row[1]],
                       model.neg_response_batch: [row[1]],
                       model.context_topic_batch: true_topic}
        )
        test_score = test_score[0]
        is_pos = evaluate_one_row(candidates_tensor, true_context, sess, model, test_score, row[1], true_topic)
        if is_pos:
            pos += 1
        else:
            neg += 1
        ind += 1
    return (pos, neg, pos / (pos+neg))


def evaluate_one_row(candidates_tensor, true_context, sess, model, test_score, true_response, true_topic):
    for batch, _ in batch_iter(candidates_tensor, 256, candidates_tensor):
        candidate_responses = batch[:, 0, :]
        context_batch = np.repeat(true_context, candidate_responses.shape[0], axis=0)
        topic_batch = np.repeat(true_topic, candidate_responses.shape[0], axis=0)

        scores = sess.run(
            model.f_pos,
            feed_dict={model.context_batch: context_batch,
                       model.response_batch: candidate_responses,
                       model.neg_response_batch: candidate_responses,
                       model.context_topic_batch: topic_batch}
        )
        for ind, score in enumerate(scores):
            if score == float('Inf') or score == -float('Inf') or score == float('NaN'):
                print(score, ind, scores[ind])
                raise ValueError
            if score >= test_score and not np.array_equal(candidate_responses[ind], true_response):
                return False
    return True


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', help='Path to test filename')
    parser.add_argument('--vocab', default='data/vocab.tsv')
    parser.add_argument('--test_topic', help='Path to test topic filename')
    parser.add_argument('--vocab_topic')
    parser.add_argument('--candidates', default='data/candidates.tsv')
    parser.add_argument('--checkpoint_dir')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--margin', type=float, default=0.01)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = _parse_args()
    vocab = load_vocab(args.vocab)
    test_tensor = make_tensor(args.test, vocab)
    vocab_topic = load_vocab(args.vocab_topic)
    test_topic = make_tensor(args.test_topic, vocab_topic)
    candidates_tensor = make_tensor(args.candidates, vocab)
    model = Model(len(vocab), emb_dim=args.emb_dim, margin=args.margin, vocab_topic_dim=len(vocab_topic))
    main(test_tensor, candidates_tensor, model, args.checkpoint_dir, test_topic)
