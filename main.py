import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import math


def score(context, response):
    return random.random()


def best_response(context, candidates):
    index = np.argmax([score(context, response) for response in candidates])
    return candidates[index]


def parse_dialogs(filename):
    dialogs = []
    with open(filename, 'r') as f:
        dialog = []
        for line in f:
            if line.strip() == '':
                dialogs.append(dialog)
                dialog = []
            else:
                user_utt, bot_utt = line.strip().split('\t')
                utt_num = user_utt.split(' ')[0]
                user_utt = ' '.join(user_utt.split(' ')[1:])
                dialog.append((utt_num, user_utt, bot_utt))
    return dialogs


def parse_candidates(filename):
    with open(filename, 'r') as f:
        return [' '.join(line.strip().split(' ')[1:]) for line in f]


def responses_accuracy(dialogs, candidates):
    correct = 0
    count = 0
    for dialog in dialogs:
        for _, user_utt, bot_utt in dialog:
            count += 1
            context = user_utt
            response = best_response(context, candidates)
            if response == bot_utt:
                correct += 1
    return correct / count, correct, count


def build_vocab_to_ind_map(dialogs):
    vocab = set()
    for d in dialogs:
        for _, user_utt, bot_utt in d:
            vocab = vocab.union(user_utt.split(' ') + bot_utt.split(' '))
    vocab = sorted(list(vocab))

    cntr = 0
    vocab_ind_map = {}
    for w in vocab:
        vocab_ind_map[w] = cntr
        cntr += 1
    return vocab_ind_map


def build_vec(vocab_ind_map, utt):
    vocab_len = len(vocab_ind_map.keys())
    vec = np.zeros((vocab_len, 1))
    for w in utt.split(' '):
        vec[vocab_ind_map[w]] += 1
    return vec


def get_vec_set(dialogs, vocab_ind_map):
    vec_set = []
    for d in dialogs:
        for _, user_utt, bot_utt in d:
            x = build_vec(vocab_ind_map, user_utt)
            y = build_vec(vocab_ind_map, bot_utt)
            vec_set.append([x, y])
    return vec_set


def main():
    train_set_task1_dialogs = parse_dialogs('dataset/dialog-bAbI-tasks/dialog-babi-task1-API-calls-trn.txt')
    dev_set_task1_dialogs = parse_dialogs('dataset/dialog-bAbI-tasks/dialog-babi-task1-API-calls-dev.txt')
    candidates = parse_candidates('dataset/dialog-bAbI-tasks/dialog-babi-candidates.txt')
    vocab_to_ind_map = build_vocab_to_ind_map(train_set_task1_dialogs)
    vec_set = get_vec_set(train_set_task1_dialogs, vocab_to_ind_map)

    D = 32
    V = len(vocab_to_ind_map)

    context = tf.placeholder(dtype=tf.float32, name='Context', shape=[V, 1])
    response = tf.placeholder(dtype=tf.float32, name='Response', shape=[V, 1])
    # f_neg = tf.placeholder(dtype=tf.float32, name='f_neg', shape=None)
    A_var = tf.Variable(initial_value=tf.truncated_normal(shape=[D, V], stddev=1 / math.sqrt(D)))
    B_var = tf.Variable(initial_value=tf.truncated_normal(shape=[D, V], stddev=1 / math.sqrt(D)))

    resp_mult = tf.matmul(B_var, response)
    cont_mult = tf.matmul(A_var, context)

    f = tf.matmul(tf.transpose(cont_mult), resp_mult)

    m = 0.01
    loss = tf.nn.relu(f)

    LR = 0.01
    optimizer = tf.train.GradientDescentOptimizer(LR).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('log/my_graph', sess.graph)
        avg_loss = 0.0
        for _ in tqdm(range(10)):
            for x, y in tqdm(vec_set):
                y_negs = random.sample(vec_set, 1)
                for _, y_neg in y_negs:
                    f_neg_val = sess.run([f], feed_dict={context: x, response: y_neg})
                    print('HERE', f_neg_val)
                    _, loss_val = sess.run([optimizer, loss], feed_dict={context: x, response: y_neg})
                    avg_loss += loss[0][0]
            avg_loss = avg_loss / len(vec_set)
            print(los, avg_loss)
            val_pos = sess.run([f], feed_dict={context: vec_set[-1][0], response: vec_set[-1][1]})
            val_neg = sess.run([f], feed_dict={context: vec_set[-1][0], response: vec_set[33][1]})
            print(val_pos, val_neg)


if __name__ == '__main__':
    main()
