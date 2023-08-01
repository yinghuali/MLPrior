import joblib
import numpy as np
import pandas as pd
import argparse
from utils import *
from get_rank_idx import *
from xgboost import XGBClassifier
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


ap = argparse.ArgumentParser()
ap.add_argument("--path_data", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--n_mutants", type=int)
ap.add_argument("--ratio_mutation_node", type=float)
ap.add_argument("--path_target_model", type=str)
ap.add_argument("--mutation_cols_level", type=int)
ap.add_argument("--n_mutants_data", type=int)
ap.add_argument("--label_name", type=str)

args = ap.parse_args()

path_data = args.path_data
model_name = args.model_name
n_mutants = args.n_mutants
mutation_level = args.ratio_mutation_node
path_target_model = args.path_target_model
mutation_cols_level = args.mutation_cols_level
n_mutants_data = args.n_mutants_data
label_name = args.label_name

mutation_cols_level = list(range(1, mutation_cols_level))
data_name = path_data.split('/')[-2]+'_'+path_data.split('/')[-1].split('.')[0]
sava_path_subject_model_name = 'result/'+model_name+'_'+data_name+'_model'
sava_path_subject_compare_name = 'result/'+model_name+'_'+data_name+'_compare.csv'

x, y = read_data(path_data, label_name)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


def get_mutation_Decision_tree_model_x(n_mutants, ratio_mutation_node, path_target_model, x_test, x_train):
    model_pre_test_np = []
    model_pre_train_np = []
    for _ in range(n_mutants):
        model = joblib.load(path_target_model)
        nodes_list = list(range(model.tree_.node_count))
        select_nodes_list = random.sample(nodes_list, int(len(nodes_list) * ratio_mutation_node))
        for i in select_nodes_list:
            if model.tree_.children_left[i] != model.tree_.children_right[i]:
                model.tree_.threshold[i] = model.tree_.threshold[i] * random.uniform(0, 1)
        y_test_pre = model.predict(x_test)
        y_train_pre = model.predict(x_train)

        model_pre_test_np.append(y_test_pre)
        model_pre_train_np.append(y_train_pre)

    return model_pre_train_np, model_pre_test_np


# Feature1: original feature
target_model = joblib.load(path_target_model)
target_test_pre = target_model.predict(x_test)
target_train_pre = target_model.predict(x_train)

model_pre_train_np, model_pre_test_np = get_mutation_Decision_tree_model_x(n_mutants, mutation_level, path_target_model, x_test, x_train)

# Feature2: mutation model feature
mutation_model_feature_test_vec = get_mutation_feature(model_pre_test_np, target_test_pre)
mutation_model_feature_train_vec = get_mutation_feature(model_pre_train_np, target_train_pre)

# Feature3: mutation feature
target_pre = target_model.predict(x)
mutation_x_np = get_mutation_data(x, mutation_cols_level, n_mutants_data)
mutation_x_pre_np = np.array([target_model.predict(i) for i in mutation_x_np])
mutation_x_feature = get_mutation_feature(mutation_x_pre_np, target_pre)
mutation_x_train_feature, mutation_x_test_feature, mutation_y_train, mutation_y_test = train_test_split(mutation_x_feature, y, test_size=0.3, random_state=0)

concat_train_all_feature = np.hstack((x_train, mutation_model_feature_train_vec, mutation_x_train_feature))
concat_test_all_feature = np.hstack((x_test, mutation_model_feature_test_vec, mutation_x_test_feature))

miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, y_train, y_test)


def get_model_apfd(Model, feature_train, feature_test, dt=False):
    if dt:
        model = DecisionTreeClassifier(min_samples_leaf=10)
    else:
        model = Model()
    model.fit(feature_train, miss_train_label)
    feature_pre = model.predict_proba(feature_test)[:, 1]
    feature_rank_idx = feature_pre.argsort()[::-1].copy()
    feature_apfd = apfd(idx_miss_test_list, feature_rank_idx)
    return feature_apfd


def main():
    max_depth_list = [1, 3, 5, 7, 9]
    colsample_bytree_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    learning_rate_list = [0.001, 0.01, 0.05, 0.1, 0.5]
    dic_depth = {}
    dic_colsample = {}
    dic_learning = {}
    for i in max_depth_list:
        model = XGBClassifier(max_depth=i)
        model.fit(concat_train_all_feature, miss_train_label)
        y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
        xgb_rank_idx = y_concat_all.argsort()[::-1].copy()
        xgb_apfd = apfd(idx_miss_test_list, xgb_rank_idx)
        print(i, xgb_apfd)
        dic_depth[i] = xgb_apfd
    json.dump(dic_depth, open(sava_path_subject_model_name + '_dic_depth.json', 'w'), sort_keys=False, indent=4)

    for i in colsample_bytree_list:
        model = XGBClassifier(colsample_bytree=i)
        model.fit(concat_train_all_feature, miss_train_label)
        y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
        xgb_rank_idx = y_concat_all.argsort()[::-1].copy()
        xgb_apfd = apfd(idx_miss_test_list, xgb_rank_idx)
        print(i, xgb_apfd)
        dic_colsample[i] = xgb_apfd
    json.dump(dic_colsample, open(sava_path_subject_model_name + '_dic_colsample.json', 'w'), sort_keys=False, indent=4)

    for i in learning_rate_list:
        model = XGBClassifier(learning_rate=i)
        model.fit(concat_train_all_feature, miss_train_label)
        y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
        xgb_rank_idx = y_concat_all.argsort()[::-1].copy()
        xgb_apfd = apfd(idx_miss_test_list, xgb_rank_idx)
        print(i, xgb_apfd)
        dic_learning[i] = xgb_apfd
    json.dump(dic_learning, open(sava_path_subject_model_name+'_dic_learning.json', 'w'), sort_keys=False, indent=4)


if __name__ == '__main__':
    main()

