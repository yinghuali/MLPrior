import joblib
import pandas as pd
import numpy as np
import argparse
from utils import *
from get_rank_idx import *
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--path_data", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--n_mutants", type=int)
ap.add_argument("--path_target_model", type=str)
ap.add_argument("--mutation_cols_level", type=int)
ap.add_argument("--n_mutants_data", type=int)
ap.add_argument("--label_name", type=str)

args = ap.parse_args()

path_data = args.path_data
model_name = args.model_name
n_mutants = args.n_mutants
path_target_model = args.path_target_model
mutation_cols_level = args.mutation_cols_level
n_mutants_data = args.n_mutants_data
label_name = args.label_name

mutation_cols_level = list(range(1, mutation_cols_level))
data_name = path_data.split('/')[-2]+'_'+path_data.split('/')[-1].split('.')[0]
sava_path_subject_model_name = 'result/'+model_name+'_'+data_name+'_model.csv'
sava_path_subject_compare_name = 'result/'+model_name+'_'+data_name+'_compare.csv'

x, y = read_data(path_data, label_name)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


def get_mutation_NB_model_x(n_mutants, path_target_model, x_test, x_train):
    model_pre_test_np = []
    model_pre_train_np = []
    for _ in range(n_mutants):
        model = joblib.load(path_target_model)
        pre_test_np = model.predict_proba(x_test)[:, 1]
        pre_train_np = model.predict_proba(x_train)[:, 1]
        threshold_value = random.uniform(0.3, 0.8)

        y_test_pre = []
        y_train_pre = []
        for i in pre_test_np:
            if i >= threshold_value:
                y_test_pre.append(1)
            else:
                y_test_pre.append(0)
        for i in pre_train_np:
            if i >= threshold_value:
                y_train_pre.append(1)
            else:
                y_train_pre.append(0)

        y_test_pre = model.predict(x_test)
        y_train_pre = model.predict(x_train)

        model_pre_test_np.append(y_test_pre)
        model_pre_train_np.append(y_train_pre)

    return model_pre_train_np, model_pre_test_np


# Feature1: original feature
target_model = joblib.load(path_target_model)
target_test_pre = target_model.predict(x_test)
target_train_pre = target_model.predict(x_train)

model_pre_train_np, model_pre_test_np = get_mutation_NB_model_x(n_mutants, path_target_model, x_test, x_train)


# Feature2: mutation model feature
mutation_model_feature_test_vec = get_mutation_feature(model_pre_test_np, target_test_pre)
mutation_model_feature_train_vec = get_mutation_feature(model_pre_train_np, target_train_pre)


# Feature3: mutation feature
target_pre = target_model.predict(x)
mutation_x_np = get_mutation_data(x, mutation_cols_level, n_mutants_data)
mutation_x_pre_np = np.array([target_model.predict(i) for i in mutation_x_np])
mutation_x_feature = get_mutation_feature(mutation_x_pre_np, target_pre)
mutation_x_train_feature, mutation_x_test_feature, mutation_y_train, mutation_y_test = train_test_split(mutation_x_feature, y, test_size=0.3, random_state=0)

fusion_3_feature_train = np.hstack((x_train, mutation_model_feature_train_vec, mutation_x_train_feature))
fusion_3_feature_test = np.hstack((x_test, mutation_model_feature_test_vec, mutation_x_test_feature))
miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, y_train, y_test)


def get_model_apfd(Model, dt=False):
    if dt:
        model = DecisionTreeClassifier(min_samples_leaf=10)
    else:
        model = Model()
    model.fit(fusion_3_feature_train, miss_train_label)
    feature_pre = model.predict_proba(fusion_3_feature_test)[:, 1]
    fusion_3_feature_rank_idx = feature_pre.argsort()[::-1].copy()
    fusion_3_feature_apfd = apfd(idx_miss_test_list, fusion_3_feature_rank_idx)

    res_list = [fusion_3_feature_apfd]
    return res_list


def get_compare_method_apfd(target_model, x_test):
    x_test_target_model_pre = target_model.predict_proba(x_test)

    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(x_test_target_model_pre)
    pcs_rank_idx = PCS_rank_idx(x_test_target_model_pre)
    entropy_rank_idx = Entropy_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)

    random_apfd = apfd(idx_miss_test_list, random_rank_idx)
    deepGini_apfd = apfd(idx_miss_test_list, deepGini_rank_idx)
    vanillasoftmax_apfd = apfd(idx_miss_test_list, vanillasoftmax_rank_idx)
    pcs_apfd = apfd(idx_miss_test_list, pcs_rank_idx)
    entropy_apfd = apfd(idx_miss_test_list, entropy_rank_idx)

    res_list = [random_apfd, deepGini_apfd, vanillasoftmax_apfd, pcs_apfd, entropy_apfd]

    return res_list


def main():
    lr_res = ['lr'] + get_model_apfd(LogisticRegression, dt=False)
    dt_res = ['dt'] + get_model_apfd(DecisionTreeClassifier, dt=True)
    xgb_res = ['xgb'] + get_model_apfd(XGBClassifier, dt=False)
    nb_res = ['nb'] + get_model_apfd(GaussianNB, dt=False)
    knn_res = ['knn'] + get_model_apfd(GaussianNB, dt=False)
    df_model = pd.DataFrame([lr_res, dt_res, xgb_res, nb_res, knn_res], columns=['Approach', 'apfd'])
    df_model.to_csv(sava_path_subject_model_name, index=False)
    res_list = get_compare_method_apfd(target_model, x_test)
    Approach_list = ['random_apfd', 'deepGini_apfd', 'vanillasoftmax_apfd', 'pcs_apfd', 'entropy_apfd']
    df_compare = pd.DataFrame(columns=['Approach'])
    df_compare['Approach'] = Approach_list
    df_compare['apfd'] = res_list
    df_compare.to_csv(sava_path_subject_compare_name, index=False)



if __name__ == '__main__':
    main()




