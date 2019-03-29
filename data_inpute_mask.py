import numpy as np
import h5py

def inpute(path, split):
    if split == 'train':
        h5file = h5py.File(path + '.h5', 'r')
        read_data = h5file['train_data'][...]#得到的是一个HDF5的dataset，并不是array
        read_labels = h5file['train_labels'][...]
        read_mask = h5file['train_mask'][...]
        read_data = np.array(read_data, dtype=np.float32)
        read_labels = np.array(read_labels, dtype=np.float32)
        read_mask = np.array(read_mask, dtype=np.float32)
        return read_data, read_labels, read_mask

    elif split == 'test':
        h5file = h5py.File(path + '.h5', 'r')
        read_data = h5file['test_data'][...]
        read_labels = h5file['test_labels'][...]
        return read_data, read_labels


def process(read_data, read_labels, read_mask):
    batch = 16                                                                                      #每次输入图片个数
    size = 64

    data_cases, data_height, data_width, data_channels = read_data.shape
    labels_cases, labels_height, labels_width, labels_channels = read_labels.shape
    mask_cases, mask_height, mask_width, mask_channels = read_mask.shape
    rand_index = np.random.random_integers(0, data_cases - 1, size=batch)
    rand_index.sort()
    data = read_data[rand_index, :, :, :]
    labels = read_labels[rand_index, :, :, :]
    mask = read_mask[rand_index, :, :, :]
    crops_x = np.random.random_integers(0, high=data_height - size, size=batch)
    crops_y = np.random.random_integers(0, high=data_width - size, size=batch)
    random_cropped_data = np.zeros((batch, size, size, data_channels), dtype=np.float32)
    random_cropped_labels = np.zeros((batch, size, size, labels_channels), dtype=np.float32)
    random_cropped_mask = np.zeros((batch, size, size, labels_channels), dtype=np.float32)

    for i in range(batch):
        random_cropped_data[i, :, :, :] = data[i, crops_x[i]: (crops_x[i] + size), crops_y[i]: (crops_y[i] + size), :]
        random_cropped_labels[i, :, :, :] = labels[i, (crops_x[i]):(crops_x[i] + size), (crops_y[i]):(crops_y[i] + size), :]
        random_cropped_mask[i, :, :, :] = mask[i, crops_x[i]: (crops_x[i] + size), crops_y[i]: (crops_y[i] + size), :]

    return random_cropped_data, random_cropped_labels, random_cropped_mask


def inpute_test(path):
    h5file = h5py.File(path + '.h5', 'r')
    read_data = h5file['train_data'][...]
    read_data = np.array(read_data, dtype=np.float32)

    return read_data


