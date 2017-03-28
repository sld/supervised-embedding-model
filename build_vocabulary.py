from signal import signal, SIGPIPE, SIG_DFL
from sys import stdin

if __name__ == '__main__':
    signal(SIGPIPE, SIG_DFL)

    fin = stdin
    vocab = set()
    for line in fin:
        context, response = line.strip().split('\t')
        for w in context.split(' '):
            vocab.add(w)

        for w in response.split(' '):
            vocab.add(w)

    vocab = list(vocab)
    for i in range(len(vocab)):
        print('{}\t{}'.format(i, vocab[i]))
