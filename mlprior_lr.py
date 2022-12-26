import joblib
import numpy as np
from utils import *
from get_rank_idx import *
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

path_data = 'data/adult.csv'
model_name = 'lr'
n_mutants = 20                # number of mutant models
mutation_level = [2, 10]      # range of mutation models
path_target_model = 'models/target_models/lr.model'
mutation_cols_level = [1, 5]  # range of mutation cols
n_mutants_data = 20           # number of mutation data


data_name = path_data.split('/')[-1].split('.')[0]


def get_data(data_name):
    if data_name=='adult':
        x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_ = get_adult_x_y(path_data)
        return x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test

    else:
        print('please input data name')


def get_mutation_model_x(n_mutants, mutation_level, path_target_model):
    model_pre_test_np = []
    model_pre_train_np = []
    for _ in range(n_mutants):
        model = joblib.load(path_target_model)
        for i in range(len(model.coef_[0])):
            ratio = random.randint(mutation_level[0], mutation_level[1])
            model.coef_[0][i] = model.coef_[0][i]*ratio
        y_test_pre = model.predict(x_normalization_test)
        y_train_pre = model.predict(x_normalization_train)

        model_pre_test_np.append(y_test_pre)
        model_pre_train_np.append(y_train_pre)

    model_pre_test_np = np.array(model_pre_test_np)
    model_pre_train_np = np.array(model_pre_train_np)

    return model_pre_train_np, model_pre_test_np


# Feature1: original feature
x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_ = get_data(data_name)
target_model = joblib.load(path_target_model)
target_test_pre = target_model.predict(x_normalization_test)
target_train_pre = target_model.predict(x_normalization_train)

model_pre_train_np, model_pre_test_np = get_mutation_model_x(n_mutants, mutation_level, path_target_model)

# Feature2: mutation model feature
mutation_model_feature_test_vec = get_mutation_feature(model_pre_test_np, target_test_pre)
mutation_model_feature_train_vec = get_mutation_feature(model_pre_train_np, target_train_pre)


# Feature3: mutation of original feature
x, x_normalization, y = get_adult_x_y_all(path_data)
target_pre = target_model.predict(x_normalization)
mutation_x_np = get_mutation_data(x_normalization, mutation_cols_level, n_mutants_data)
mutation_x_pre_np = np.array([target_model.predict(i) for i in mutation_x_np])
mutation_x_feature = get_mutation_feature(mutation_x_pre_np, target_pre)
mutation_x_train_feature, mutation_x_test_feature, mutation_y_train, mutation_y_test = train_test_split(mutation_x_feature, y, test_size=0.3, random_state=17)

miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, y_train, y_test)

#####
rf = RandomForestClassifier()
rf.fit(mutation_x_train_feature, miss_train_label)
feature_pre = rf.predict_proba(mutation_x_test_feature)[:, 1]
mutation_feature_rank_idx = feature_pre.argsort()[::-1].copy()
#####

#####
rf = RandomForestClassifier()
rf.fit(mutation_model_feature_train_vec, miss_train_label)
feature_pre = rf.predict_proba(mutation_model_feature_test_vec)[:, 1]
mutation_model_rank_idx = feature_pre.argsort()[::-1].copy()
######

########
rf = RandomForestClassifier()
rf.fit(x_train, miss_train_label)
feature_pre = rf.predict_proba(x_test)[:, 1]
feature_rank_idx = feature_pre.argsort()[::-1].copy()
########

########
fusion_2_feature_train = np.hstack((mutation_model_feature_train_vec, mutation_x_train_feature))
fusion_2_feature_test = np.hstack((mutation_model_feature_test_vec, mutation_x_test_feature))
rf = RandomForestClassifier()
rf.fit(fusion_2_feature_train, miss_train_label)
feature_pre = rf.predict_proba(fusion_2_feature_test)[:, 1]
fusion_2_feature_rank_idx = feature_pre.argsort()[::-1].copy()
########


########
fusion_3_feature_train = np.hstack((x_train, mutation_model_feature_train_vec, mutation_x_train_feature))
fusion_3_feature_test = np.hstack((x_test, mutation_model_feature_test_vec, mutation_x_test_feature))
rf = RandomForestClassifier()
rf.fit(fusion_3_feature_train, miss_train_label)
feature_pre = rf.predict_proba(fusion_3_feature_test)[:, 1]
fusion_3_feature_rank_idx = feature_pre.argsort()[::-1].copy()
########

mutation_feature_apfd = [apfd(idx_miss_test_list, mutation_feature_rank_idx)]
print('mutation_feature_apfd', mutation_feature_apfd)

mutation_model_apfd = [apfd(idx_miss_test_list, mutation_model_rank_idx)]
print('mutation_model_apfd', mutation_model_apfd)

original_feature_apfd = [apfd(idx_miss_test_list, feature_rank_idx)]
print('original_feature_apfd', original_feature_apfd)

fusion_2_feature_apfd = [apfd(idx_miss_test_list, fusion_2_feature_rank_idx)]
print('fusion_2_feature_apfd', fusion_2_feature_apfd)

fusion_3_feature_apfd = [apfd(idx_miss_test_list, fusion_3_feature_rank_idx)]
print('fusion_3_feature_apfd', fusion_3_feature_apfd)


x_test_target_model_pre = target_model.predict_proba(x_normalization_test)

margin_rank_idx = Margin_rank_idx(x_test_target_model_pre)
deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
leastConfidence_rank_idx = LeastConfidence_rank_idx(x_test_target_model_pre)
random_rank_idx = Random_rank_idx(x_test_target_model_pre)

random_apfd = [apfd(idx_miss_test_list, random_rank_idx)]
deepGini_apfd = [apfd(idx_miss_test_list, deepGini_rank_idx)]
leastConfidence_apfd = [apfd(idx_miss_test_list, leastConfidence_rank_idx)]
margin_apfd = [apfd(idx_miss_test_list, margin_rank_idx)]

print('random_apfd', random_apfd)
print('deepGini_apfd', deepGini_apfd)
print('leastConfidence_apfd', leastConfidence_apfd)
print('margin_apfd', margin_apfd)




