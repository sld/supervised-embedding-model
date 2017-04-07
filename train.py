import tensorflow as tf
import numpy as np
from tqdm import tqdm
from make_train_tensor import make_tensor, load_vocab
from model import Model
from sys import argv
from test import evaluate
from utils import batch_iter


def main(train_tensor, dev_tensor, candidates_tensor, model):
    train_steps = 40
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(train_steps):
            avg_loss = 0
            for batch in batch_iter(train_tensor, 256, True):
                # TODO: Make f_neg in the loop
                f_neg = sess.run(
                    model.f,
                    feed_dict={model.context_batch: batch[0][0], model.response_batch: batch[1][1]}
                )

                loss = sess.run(
                    [model.loss, model.optimizer],
                    feed_dict={model.context_batch: batch[0][0],
                               model.response_batch: batch[0][1],
                               model.f_neg: f_neg}
                )
                avg_loss += loss[0]
            avg_loss = avg_loss / train_tensor.shape[0]
            avg_dev_loss = 0
            for batch in batch_iter(dev_tensor, 128):
                f_neg = sess.run(
                    model.f,
                    feed_dict={model.context_batch: batch[0][0], model.response_batch: batch[1][1]}
                )

                loss = sess.run(
                    [model.loss],
                    feed_dict={model.context_batch: batch[0][0],
                               model.response_batch: batch[0][1],
                               model.f_neg: f_neg}
                )
                avg_dev_loss += loss[0]
            avg_dev_loss = avg_dev_loss / dev_tensor.shape[0]
            dev_eval = evaluate(dev_tensor, candidates_tensor, sess, model)

            print('Epoch: {}; Train loss: {}; Dev loss: {}; Eval scores: {}'.format(
                epoch, avg_loss, avg_dev_loss, dev_eval)
            )


if __name__ == '__main__':
    train_filename = argv[1]
    vocab_filename = argv[2]
    dev_filename = argv[3]
    candidates_filename = argv[4]
    vocab = load_vocab(vocab_filename)
    train_tensor = make_tensor(train_filename, vocab_filename)
    dev_tensor = make_tensor(dev_filename, vocab_filename)
    candidates_tensor = make_tensor(candidates_filename, vocab_filename)
    model = Model(len(vocab), 32)
    main(train_tensor, dev_tensor, candidates_tensor, model)
