import tensorflow as tf
import numpy as np
from tqdm import tqdm
from make_train_tensor import make_tensor, load_vocab
from model import Model
from sys import argv


def batch_iter(tensor, batch_size, shuffle=False):
    batches_count = tensor.shape[0] // batch_size

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
        data = tensor[shuffle_indices]
    else:
        data = tensor

    neg_shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
    negative_data = tensor[neg_shuffle_indices]

    for batch_num in range(batches_count):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1)*batch_size, tensor.shape[0])
        yield (data[start_index:end_index], negative_data[start_index:end_index])


def main(train_tensor, model):
    train_steps = 5000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in tqdm(range(train_steps)):
            avg_loss = 0
            for batch in batch_iter(train_tensor, 64, True):
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
            print(avg_loss)                


if __name__ == '__main__':
    train_filename = argv[1]
    vocab_filename = argv[2]
    vocab = load_vocab(vocab_filename)
    train_tensor = make_tensor(train_filename, vocab_filename)
    model = Model(len(vocab), 32)
    main(train_tensor, model)
