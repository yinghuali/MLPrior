import joblib
import pandas as pd
import argparse
from utils import *
from get_rank_idx import *
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--path_data", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--n_mutants", type=int)
ap.add_argument("--mutation_level", type=int)
ap.add_argument("--path_target_model", type=str)
ap.add_argument("--mutation_cols_level", type=int)
ap.add_argument("--n_mutants_data", type=int)
ap.add_argument("--label_name", type=str)

args = ap.parse_args()

path_data = args.path_data
model_name = args.model_name
n_mutants = args.n_mutants
mutation_level = args.mutation_level
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


def get_mutation_KNN_model_x(n_mutants, mutation_level, path_target_model, x_test, x_train):
    """mutation_level > 10"""
    model_pre_test_np = []
    model_pre_train_np = []
    for _ in range(n_mutants):
        model = joblib.load(path_target_model)
        model.n_neighbors = random.randint(10, mutation_level)
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

model_pre_train_np, model_pre_test_np = get_mutation_KNN_model_x(n_mutants, mutation_level, path_target_model, x_test, x_train)

# Feature2: mutation model feature
mutation_model_feature_test_vec = get_mutation_feature(model_pre_test_np, target_test_pre)
mutation_model_feature_train_vec = get_mutation_feature(model_pre_train_np, target_train_pre)


# Feature3: mutation feature
target_pre = target_model.predict(x)
mutation_x_np = get_mutation_data(x, mutation_cols_level, n_mutants_data)
mutation_x_pre_np = np.array([target_model.predict(i) for i in mutation_x_np])
mutation_x_feature = get_mutation_feature(mutation_x_pre_np, target_pre)
mutation_x_train_feature, mutation_x_test_feature, mutation_y_train, mutation_y_test = train_test_split(mutation_x_feature, y, test_size=0.3, random_state=0)

feature_train_wo_original = np.hstack((mutation_model_feature_train_vec, mutation_x_train_feature))
feature_test_wo_original = np.hstack((mutation_model_feature_test_vec, mutation_x_test_feature))

feature_train_wo_mutation_model_feature = np.hstack((x_train, mutation_x_train_feature))
feature_test_wo_mutation_model_feature = np.hstack((x_test, mutation_x_test_feature))

feature_train_wo_mutation_feature = np.hstack((x_train, mutation_model_feature_train_vec))
feature_test_wo_mutation_feature = np.hstack((x_test, mutation_model_feature_test_vec))

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
    apfd_wo_original = get_model_apfd(XGBClassifier, feature_train_wo_original, feature_test_wo_original, dt=False)
    apfd_wo_mutation_model_feature = get_model_apfd(XGBClassifier, feature_train_wo_mutation_model_feature, feature_test_wo_mutation_model_feature, dt=False)
    apfd_wo_mutation_feature = get_model_apfd(XGBClassifier, feature_train_wo_mutation_feature, feature_test_wo_mutation_feature, dt=False)
    df = pd.DataFrame([[apfd_wo_original, apfd_wo_mutation_model_feature, apfd_wo_mutation_feature]], columns=['apfd_wo_original', 'apfd_wo_mutation_model_feature', 'apfd_wo_mutation_feature'])
    df.to_csv(sava_path_subject_model_name, index=False)


if __name__ == '__main__':
    main()


