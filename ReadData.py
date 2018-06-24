import numpy as np
import os

batch_size = 60
batch_interval = 30


def get_type(name: str):
    index = name.find("a")
    try:
        int(name[index + 2:index + 3])
        return int(name[index + 1:index + 3])
    except ValueError:
        return int(name[index + 1:index + 2])


def read_data(path: str):
    # label array: 1-dim array, every num reflects to 60 lines in action array.
    label_temp = np.array([1])
    # action array: 60 lines for each action data,
    skeleton_temp = np.ones(shape=(1, 36), dtype=np.float32)
    for root, dirs, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            file_data = np.loadtxt(file_path, np.float32)
            for batch in range(batch_size, file_data.shape[0], batch_interval):
                # label
                label_temp = np.append(label_temp, get_type(name))
                # data
                skeleton_temp = np.concatenate((skeleton_temp, file_data[batch - batch_size:batch, :]), axis=0)
    labels = label_temp[1:]
    skeleton = skeleton_temp[1:, :]
    return skeleton, labels
