import numpy as np


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
