import numpy as np


def batch_iter(tensor, batch_size, topic_tensor, shuffle=False):
    batches_count = tensor.shape[0] // batch_size

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
        data = tensor[shuffle_indices]
        data_topic = topic_tensor[shuffle_indices]
    else:
        data = tensor
        data_topic = topic_tensor

    for batch_num in range(batches_count):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1)*batch_size, tensor.shape[0])
        yield data[start_index:end_index], data_topic[start_index:end_index]


def neg_sampling_iter(tensor, batch_size, count, seed=None):
    batches_count = tensor.shape[0] // batch_size
    trials = 0
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(tensor.shape[0]))
    data = tensor[shuffle_indices]
    for batch_num in range(batches_count):
        trials += 1
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1)*batch_size, tensor.shape[0])
        if trials > count:
            return
        else:
            yield data[start_index:end_index]
