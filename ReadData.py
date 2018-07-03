import random

import numpy as np
import os

batch_size = 60
batch_interval = 30


def get_type(name):
    index = name.find("a")
    try:
        int(name[index + 2:index + 3])
        return int(name[index + 1:index + 3])
    except ValueError:
        return int(name[index + 1:index + 2])


def read_data(path):
    if os.path.exists(path + "/skeleton.npy") and os.path.exists(path + "/labels.npy"):
        labels = np.load(path + "/labels.npy")
        labels = np.eye(np.max(labels) + 1)[labels]
        return np.load(path + "/skeleton.npy"), labels

    # label array: 1-dim array, every num reflects to 60 lines in action array. [-1]
    label_temp = np.array([1])
    # action array: 60 lines for each action data. [-1, 60, 36]
    skeleton_temp = np.ones(shape=(1, 36), dtype=np.float32)

    # 1 3 9 12
    files = os.walk(path)
    random.shuffle(files)
    for root, dirs, files in files:
        for index, name in enumerate(files):
            file_path = os.path.join(root, name)
            print("file count: " + str(index) + " / " + str(len(files)))
            file_data = np.loadtxt(file_path, np.float32)
            for batch in range(batch_size, file_data.shape[0], batch_interval):
                # label
                type_num = get_type(name)
                if type_num == 1:
                    label_temp = np.append(label_temp, 0)
                elif type_num == 3:
                    label_temp = np.append(label_temp, 1)
                elif type_num == 9:
                    label_temp = np.append(label_temp, 2)
                elif type_num == 12:
                    label_temp = np.append(label_temp, 3)
                else:
                    print("unexpected label type " + name)
                    exit(1)
                # data
                skeleton_temp = np.concatenate((skeleton_temp, file_data[batch - batch_size:batch, :]), axis=0)
    labels = label_temp[1:]
    skeleton = skeleton_temp[1:, :]
    skeleton = skeleton.reshape([-1, 60, 36])
    np.save(path + "/skeleton.npy", skeleton)
    np.save(path + "/labels.npy", labels)
    labels = np.eye(np.max(labels) + 1)[labels]
    return skeleton, labels
