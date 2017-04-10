import tensorflow as tf
import numpy as np
from tqdm import tqdm
from make_train_tensor import make_tensor, load_vocab
from model import Model
from sys import argv
from test import evaluate
from utils import batch_iter, neg_sampling_iter


def main(train_tensor, dev_tensor, candidates_tensor, model, name='task1'):
    epochs = 400
    batch_size = 32
    neg_size = 100
    prev_best_accuracy = 0

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # initial_step = model.global_step.eval()
        # writer = tf.summary.FileWriter('log/supervised_emb_'+name, sess.graph)

        for epoch in range(epochs):
            avg_loss = 0
            for batch in batch_iter(train_tensor, batch_size, True):
                for neg_batch in neg_sampling_iter(train_tensor, batch_size, neg_size):
                    # Поботай siamese network и tf relational learn tutorial

                    loss = sess.run(
                        [model.loss, model.optimizer],
                        feed_dict={model.context_batch: batch[:, 0, :],
                                   model.response_batch: batch[:, 1, :],
                                   model.neg_response_batch: neg_batch[:, 1, :]}
                    )
                    avg_loss += loss[0]
                    # writer.add_summary(summary, global_step=epoch)
            avg_loss = avg_loss / (train_tensor.shape[0]*neg_size)
            avg_dev_loss = 0
            for batch in batch_iter(dev_tensor, 256):
                for neg_batch in neg_sampling_iter(dev_tensor, 256, 1):
                    loss = sess.run(
                        [model.loss],
                        feed_dict={model.context_batch: batch[:, 0, :],
                                   model.response_batch: batch[:, 1, :],
                                   model.neg_response_batch: neg_batch[:, 1, :]}
                    )
                    avg_dev_loss += loss[0]
            avg_dev_loss = avg_dev_loss / (dev_tensor.shape[0]*1)

            print('Epoch: {}; Train loss: {}; Dev loss: {};'.format(
                epoch, avg_loss, avg_dev_loss)
            )
            if epoch % 5 == 0:
                dev_eval = evaluate(dev_tensor, candidates_tensor, sess, model)
                print('Evaluation in dev set: {}'.format(dev_eval))
                accuracy = dev_eval[2]
                if accuracy >= prev_best_accuracy:
                    print('Saving Model')
                    prev_best_accuracy = accuracy
                    saver.save(sess, 'checkpoints/{}-best-acc'.format(name))


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
