#这个程序生成了一个h5文件，相当于一个字典，里面有train_data[350,256,256,2] train_labels[350,256,256,1] train_mask[350,256,256,1]


import scipy.io as scio
import numpy as np
import h5py
import os.path as osp
import matplotlib.pyplot as plt

this_dir = osp.dirname(__file__)

path_full = '350_m0_noise_6e7_035_16_only_T2.mat'
data = h5py.File(path_full)
tmp_data = data['inpute']
tmp_labels = data['labels']
tmp_mask = data['mask']
tmp_data = np.transpose(tmp_data, (3, 1, 2, 0))
tmp_labels = np.transpose(tmp_labels, (2, 0, 1))
tmp_mask = np.transpose(tmp_mask, (2, 0, 1))
test_data = np.zeros(tmp_data.shape)
test_labels = np.zeros((350, 256, 256, 1), dtype=np.float32)
test_mask = np.zeros((350, 256, 256, 1), dtype=np.float32)
print(test_data.shape)
for i in range(tmp_data.shape[0]):
    test_data[i, :, :, 0] = tmp_data[i, :, :, 0]
    test_data[i, :, :, 1] = tmp_data[i, :, :, 1]

    test_labels[i, :, :, 0] = tmp_labels[i, :, :]

    test_mask[i, :, :, 0] = tmp_mask[i, :, :]

plt.figure(1)
plt.subplot(221)
plt.imshow((tmp_data[10, :, :, 0] ** 2 + tmp_data[10, :, :, 1] ** 2) ** 0.5)
plt.title('input')
plt.subplot(222)
plt.imshow(test_labels[10, :, :, 0])
plt.title('result')
plt.show()

h5file = h5py.File('350_m0_noise_6e7_035_16_only_T2.h5', 'w')
h5file.create_dataset('train_data', data=test_data, dtype=np.float32)
h5file.create_dataset('train_labels', data=test_labels, dtype=np.float32)
h5file.create_dataset('train_mask', data=test_mask, dtype=np.float32)
print(list(h5file.keys()))
h5file.close()





