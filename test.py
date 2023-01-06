import joblib
import pandas as pd
import numpy as np
import argparse
from utils import *
from get_rank_idx import *
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

path_data = 'data/heart.csv'
model_name = 'rf'
path_target_model = 'models/target_models/heart_rf.model'
path_mutation_models = 'models/mutation_models/heart/rf'
mutation_cols_level = 5
n_mutants_data = 20
label_name = 'label'

# path_data = 'data/adult.csv'
# model_name = 'lgb'
# path_target_model = 'models/target_models/adult_lgb.model'
# path_mutation_models = 'models/mutation_models/adult/lgb'
# mutation_cols_level = 5
# n_mutants_data = 20
# label_name = 'income'

mutation_cols_level = list(range(1, mutation_cols_level))
data_name = path_data.split('/')[-1].split('.')[0]
sava_path_subject_model_name = 'result/'+model_name+'_'+data_name+'_model.csv'
sava_path_subject_compare_name = 'result/'+model_name+'_'+data_name+'_compare.csv'

x, y = read_data(path_data, label_name)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


def get_mutation_model_x(path_mutation_models, x_test, x_train):
    model_pre_test_np = []
    model_pre_train_np = []
    path_model_list = get_model_path(path_mutation_models)
    path_model_list = sorted(path_model_list)
    for i in range(len(path_model_list)):
        model = joblib.load(path_model_list[i])
        y_test_pre = model.predict(x_test)
        y_train_pre = model.predict(x_train)
        model_pre_test_np.append(y_test_pre)
        model_pre_train_np.append(y_train_pre)
    model_pre_test_np = np.array(model_pre_test_np)
    model_pre_train_np = np.array(model_pre_train_np)
    return model_pre_train_np, model_pre_test_np


# Feature1: original feature
target_model = joblib.load(path_target_model)
target_test_pre = target_model.predict(x_test)
target_train_pre = target_model.predict(x_train)

model_pre_train_np, model_pre_test_np = get_mutation_model_x(path_mutation_models, x_test, x_train)

# Feature2: mutation model feature
mutation_model_feature_test_vec = get_mutation_feature(model_pre_test_np, target_test_pre)
mutation_model_feature_train_vec = get_mutation_feature(model_pre_train_np, target_train_pre)


# Feature3: mutation of original feature
target_pre = target_model.predict(x)
mutation_x_np = get_mutation_data(x, mutation_cols_level, n_mutants_data)
mutation_x_pre_np = np.array([target_model.predict(i) for i in mutation_x_np])
mutation_x_feature = get_mutation_feature(mutation_x_pre_np, target_pre)
mutation_x_train_feature, mutation_x_test_feature, mutation_y_train, mutation_y_test = train_test_split(mutation_x_feature, y, test_size=0.3, random_state=0)

fusion_2_feature_train = np.hstack((mutation_model_feature_train_vec, mutation_x_train_feature))
fusion_2_feature_test = np.hstack((mutation_model_feature_test_vec, mutation_x_test_feature))
#
fusion_3_feature_train = np.hstack((x_train, mutation_model_feature_train_vec, mutation_x_train_feature))
fusion_3_feature_test = np.hstack((x_test, mutation_model_feature_test_vec, mutation_x_test_feature))
miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, y_train, y_test)

print(fusion_3_feature_train.shape)
print(fusion_2_feature_train.shape)

model = RandomForestClassifier()
model.fit(fusion_3_feature_train, miss_train_label)
feature_pre = model.predict_proba(fusion_3_feature_test)[:, 1]
model_rank_idx = feature_pre.argsort()[::-1].copy()
re = apfd(idx_miss_test_list, model_rank_idx)
print(re)

model = XGBClassifier()
model.fit(fusion_3_feature_train, miss_train_label)
feature_pre = model.predict_proba(fusion_3_feature_test)[:, 1]
model_rank_idx = feature_pre.argsort()[::-1].copy()
re = apfd(idx_miss_test_list, model_rank_idx)
print(re)

model = LGBMClassifier()
model.fit(fusion_3_feature_train, miss_train_label)
feature_pre = model.predict_proba(fusion_3_feature_test)[:, 1]
model_rank_idx = feature_pre.argsort()[::-1].copy()
re = apfd(idx_miss_test_list, model_rank_idx)
print(re)

# def get_model_apfd(Model):
#     model = Model()
#     model.fit(mutation_x_train_feature, miss_train_label)
#     feature_pre = model.predict_proba(mutation_x_test_feature)[:, 1]
#     mutation_feature_rank_idx = feature_pre.argsort()[::-1].copy()
#
#     model = Model()
#     model.fit(mutation_model_feature_train_vec, miss_train_label)
#     feature_pre = model.predict_proba(mutation_model_feature_test_vec)[:, 1]
#     mutation_model_rank_idx = feature_pre.argsort()[::-1].copy()
#
#     model = Model()
#     model.fit(fusion_2_feature_train, miss_train_label)
#     feature_pre = model.predict_proba(fusion_2_feature_test)[:, 1]
#     fusion_2_feature_rank_idx = feature_pre.argsort()[::-1].copy()
#
#     model = Model()
#     model.fit(fusion_3_feature_train, miss_train_label)
#     feature_pre = model.predict_proba(fusion_3_feature_test)[:, 1]
#
#     fusion_3_feature_rank_idx = feature_pre.argsort()[::-1].copy()
#     mutation_feature_apfd = apfd(idx_miss_test_list, mutation_feature_rank_idx)
#     mutation_model_apfd = apfd(idx_miss_test_list, mutation_model_rank_idx)
#     fusion_2_feature_apfd = apfd(idx_miss_test_list, fusion_2_feature_rank_idx)
#     fusion_3_feature_apfd = apfd(idx_miss_test_list, fusion_3_feature_rank_idx)
#
#     res_list = [mutation_feature_apfd, mutation_model_apfd, fusion_2_feature_apfd, fusion_3_feature_apfd]
#     return res_list


def get_compare_method_apfd(target_model, x_test):
    x_test_target_model_pre = target_model.predict_proba(x_test)
    margin_rank_idx = Margin_rank_idx(x_test_target_model_pre)
    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    leastConfidence_rank_idx = LeastConfidence_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)

    random_apfd = apfd(idx_miss_test_list, random_rank_idx)
    deepGini_apfd = apfd(idx_miss_test_list, deepGini_rank_idx)
    leastConfidence_apfd = apfd(idx_miss_test_list, leastConfidence_rank_idx)
    margin_apfd = apfd(idx_miss_test_list, margin_rank_idx)

    res_list = [random_apfd, deepGini_apfd, leastConfidence_apfd, margin_apfd]

    return res_list


def main():
    # lr_res = ['lr'] + get_model_apfd(LogisticRegression)
    # rf_res = ['rf'] + get_model_apfd(RandomForestClassifier)
    #
    # xgb_res = ['xgb'] + get_model_apfd(XGBClassifier)
    # lgb_res = ['lgb'] + get_model_apfd(LGBMClassifier)
    # df_model = pd.DataFrame([lr_res, rf_res, xgb_res, lgb_res], columns=['Approach', 'mutation_feature_apfd', 'mutation_model_apfd', 'fusion_2_feature_apfd', 'fusion_3_feature_apfd'])
    #
    # df_model.to_csv(sava_path_subject_model_name, index=False)

    res_list = get_compare_method_apfd(target_model, x_test)
    Approach_list = ['random_apfd', 'deepGini_apfd', 'leastConfidence_apfd', 'margin_apfd']
    df_compare = pd.DataFrame(columns=['Approach'])
    df_compare['Approach'] = Approach_list
    df_compare['apfd'] = res_list
    # df_compare.to_csv(sava_path_subject_compare_name, index=False)
    print(res_list)


if __name__ == '__main__':
    main()

