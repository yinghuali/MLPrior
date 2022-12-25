import joblib
import numpy as np
from utils import *
from get_rank_idx import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

path_data = 'data/adult.csv'
model_name = 'lr'

data_name = path_data.split('/')[-1].split('.')[0]


def get_adult_x_y():
    df = read_adult(path_data)
    df_normalization = get_df_normalization(df, ['income'])
    x = df.to_numpy()[:, 0:-1]
    x_normalization = df_normalization.to_numpy()[:, 0:-1]
    y = df.to_numpy()[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)
    x_normalization_train, x_normalization_test, y_train_, y_test_ = train_test_split(x_normalization, y, test_size=0.3, random_state=17)
    return x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_


x_train, x_test, y_train, y_test, x_normalization_train, x_normalization_test, y_train_, y_test_ = get_adult_x_y()


target_model = joblib.load('models/target_models/'+model_name+'.model')
y_pre = target_model.predict(x_normalization_test)
acc = accuracy_score(y_pre, y_test)
print(acc)

target_test_pre = target_model.predict(x_normalization_test)
target_train_pre = target_model.predict(x_normalization_train)


model_pre_test_np = []
model_pre_train_np = []

for _ in range(20):  # 配置参数
    model = joblib.load('models/target_models/' + model_name + '.model')
    for i in range(len(model.coef_[0])):
        ratio = random.randint(2, 10)  # 配置参数
        model.coef_[0][i] = model.coef_[0][i]*ratio
    y_test_pre = model.predict(x_normalization_test)
    y_train_pre = model.predict(x_normalization_train)

    model_pre_test_np.append(y_test_pre)
    model_pre_train_np.append(y_train_pre)

model_pre_test_np = np.array(model_pre_test_np)
model_pre_train_np = np.array(model_pre_train_np)


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

feature_test_vec = get_mutation_feature(model_pre_test_np, target_test_pre)
feature_train_vec = get_mutation_feature(model_pre_train_np, target_train_pre)

target_train_pre = target_model.predict(x_normalization_train)
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

rf = RandomForestClassifier()
rf.fit(feature_train_vec, miss_train_label)
feature_pre = rf.predict_proba(feature_test_vec)[:, 1]
mutation_rank_idx = feature_pre.argsort()[::-1].copy()


########
rf = RandomForestClassifier()
rf.fit(x_train, miss_train_label)
feature_pre = rf.predict_proba(x_test)[:, 1]
feature_rank_idx = feature_pre.argsort()[::-1].copy()
########


########
feature_mutation_train = np.hstack((x_train, feature_train_vec))
feature_mutation_test = np.hstack((x_test, feature_test_vec))
rf = RandomForestClassifier()
rf.fit(feature_mutation_train, miss_train_label)
feature_pre = rf.predict_proba(feature_mutation_test)[:, 1]
feature_mutation_rank_idx = feature_pre.argsort()[::-1].copy()
########


mutation_apfd = [apfd(idx_miss_test_list, mutation_rank_idx)]
print('mutation_apfd', mutation_apfd)

feature_apfd = [apfd(idx_miss_test_list, feature_rank_idx)]
print('feature_apfd', feature_apfd)

feature_mutation_apfd = [apfd(idx_miss_test_list, feature_mutation_rank_idx)]
print('feature_mutation_apfd', feature_mutation_apfd)


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




