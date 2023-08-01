import pandas as pd
import os
import random
import numpy as np


def apfd(error_idx_list, pri_idx_list):
    error_idx_list = list(error_idx_list)
    pri_idx_list = list(pri_idx_list)
    n = len(pri_idx_list)
    m = len(error_idx_list)
    TF_list = [pri_idx_list.index(i) for i in error_idx_list]
    apfd = 1 - sum(TF_list)*1.0 / (n*m) + 1 / (2*n)
    return apfd


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    idx_miss_list.append(i)
    return idx_miss_list


def read_data(path, lable_name):
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtypes == 'object' or col==lable_name:
            category_list = sorted(list(set(df[col])))
            dic = dict(zip(category_list, range(len(category_list))))
            df[col] = [dic[i] for i in df[col]]
    y = df[lable_name].to_numpy()
    del df[lable_name]
    for i in df.columns:
        if df[i].all() != 0:
            df[i] = (df[i] - min(df[i])) / (max(df[i]) - min(df[i]))
    x = df.to_numpy()
    return x, y


def get_mutation_feature(model_pre_np, target_pre):
    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(model_pre_np)):
            if model_pre_np[j][i] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_vec = np.array(feature_list)
    return feature_vec


def get_miss_lable(target_train_pre, target_test_pre, y_train, y_test):
    idx_miss_train_list = get_idx_miss_class(target_train_pre, y_train)
    idx_miss_test_list = get_idx_miss_class(target_test_pre, y_test)
    miss_train_label = [0]*len(y_train)
    for i in idx_miss_train_list:
        miss_train_label[i]=1
    miss_train_label = np.array(miss_train_label)

    miss_test_label = [0]*len(y_test)
    for i in idx_miss_test_list:
        miss_test_label[i]=1
    miss_test_label = np.array(miss_test_label)

    return miss_train_label, miss_test_label, idx_miss_test_list


def get_mutation_data(x, mutation_cols_level, n_mutants_data):
    res_list = []
    idx_waiting_list = list(range(len(x[0])))
    for _ in range(n_mutants_data):
        n_select_cols = random.sample(mutation_cols_level, 1)[0]
        res = np.copy(x)
        idx_list = random.sample(idx_waiting_list, n_select_cols)
        for idx in idx_list:
            res[:, idx] = 0
        res_list.append(res)
    res_np = np.array(res_list)
    return res_np


def get_model_path(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.model'):
                    model_path_list.append(file_absolute_path)
    return model_path_list