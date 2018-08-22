# -*- coding: utf-8 -*-
import random
import numpy as np
import os

data_length = 24
data_interval = 24


def get_type(name):
    index = name.find("a")
    try:
        int(name[index + 2:index + 3])
        return int(name[index + 1:index + 3])
    except ValueError:
        return int(name[index + 1:index + 2])


def read_data(path, size=0):
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
    file_list = []
    for root, dirs, file in files:
        for name in file:
            file_list.append([root, name])
    random.shuffle(file_list)
    for index, [root, name] in enumerate(file_list):
        if size and index > size:
            break
        print("file count: " + str(index) + " / " + str(len(file_list)))
        file_path = os.path.join(root, name)
        file_data = np.loadtxt(file_path, np.float32)
        for data in range(data_length, file_data.shape[0], data_interval):
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
            skeleton_temp = np.concatenate((skeleton_temp, file_data[data - data_length:data, :]), axis=0)
    labels = label_temp[1:]
    skeleton = skeleton_temp[1:, :]
    skeleton = skeleton.reshape([-1, data_length, 36])
    np.save(path + "/skeleton.npy", skeleton)
    np.save(path + "/labels.npy", labels)
    labels = np.eye(np.max(labels) + 1)[labels]
    return skeleton, labels


def read_data_norm(path, size=0):
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
    file_list = []
    for root, dirs, file in files:
        for name in file:
            file_list.append([root, name])
    random.shuffle(file_list)
    for index, [root, name] in enumerate(file_list):
        if size and index > size:
            break
        print("file count: " + str(index) + " / " + str(len(file_list)))
        file_path = os.path.join(root, name)
        file_data = np.loadtxt(file_path, np.float32)
        for data in range(data_length, file_data.shape[0], data_interval):
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
            skeleton_temp = np.concatenate((skeleton_temp, file_data[data - data_length:data, :]), axis=0)
    labels = label_temp[1:]
    skeleton = skeleton_temp[1:, :]
    length = np.sqrt(np.add(np.square(skeleton[:, 2] - skeleton[:, 0]), np.square(skeleton[:, 3] - skeleton[:, 1])))
    for l in range(length.size):
        if length[l] <= 0.001:
            length[l] = 1
    for i in range(2, skeleton.shape[1], 2):
        skeleton[:, i] = (skeleton[:, i] - skeleton[:, 0]) / length
        skeleton[:, i + 1] = (skeleton[:, i + 1] - skeleton[:, 1]) / length
    for i in skeleton:
        i[0] = 0.0
        i[1] = 0.0
    skeleton = skeleton.reshape([-1, data_length, 36])
    np.save(path + "/skeleton.npy", skeleton)
    np.save(path + "/labels.npy", labels)
    labels = np.eye(np.max(labels) + 1)[labels]
    return skeleton, labels


def read_data_test(path, size=0):
    # if os.path.exists(path + "/skeleton.npy") and os.path.exists(path + "/labels.npy"):
    #     names = np.load(path + "/labels.npy")
    #     return np.load(path + "/skeleton.npy"), names

    # label array: 1-dim array, every num reflects to 60 lines in action array. [-1]
    label_temp = np.array([1])
    # action array: 60 lines for each action data. [-1, 60, 36]
    skeleton_temp = np.ones(shape=(1, 36), dtype=np.float32)

    # 1 3 9 12
    files = os.walk(path)
    file_list = []
    for root, dirs, file in files:
        for name in file:
            file_list.append([root, name])
    random.shuffle(file_list)
    for index, [root, name] in enumerate(file_list):
        if size and index > size:
            break
        print("file count: " + str(index) + " / " + str(len(file_list)))
        file_path = os.path.join(root, name)
        file_data = np.loadtxt(file_path, np.float32)
        for data in range(data_length, file_data.shape[0], data_interval):
            # label
            label_temp = np.append(label_temp, name + '_' + str(data - data_length) + '_' + str(data))
            # data
            skeleton_temp = np.concatenate((skeleton_temp, file_data[data - data_length:data, :]), axis=0)
    names = label_temp[1:]
    skeleton = skeleton_temp[1:, :]
    skeleton = skeleton.reshape([-1, data_length, 36])
    # np.save(path + "/skeleton.npy", skeleton)
    # np.save(path + "/labels.npy", names)
    return skeleton, names


def read_single_file(path):
    # 1 3 9 12
    return np.loadtxt(path, np.float32)[np.newaxis, :]
