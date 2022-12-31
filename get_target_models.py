import joblib
import argparse
from get_rank_idx import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ap = argparse.ArgumentParser()
# ap.add_argument("--path_data", type=str)
# ap.add_argument("--label_name", type=str)
# ap.add_argument("--n_estimators", type=int)
# args = ap.parse_args()

# python get_target_models.py --path_data 'data/adult.csv' --label_name 'income' --n_estimators 3
# python get_target_models.py --path_data 'data/wine.csv' --label_name 'quality' --n_estimators 10

# path_data = args.path_data
# label_name = args.label_name
# n_estimators = args.n_estimators

path_data = 'data/adult.csv'
label_name = 'income'
n_estimators = 2

data_name = path_data.split('/')[-1].split('.')[0]
x, y = read_data(path_data, label_name)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# # adult LR
# model = LogisticRegression()
# model.fit(x_train, y_train)
# joblib.dump(model, 'models/target_models/{}_lr.model'.format(data_name))
# y_pre = model.predict(x_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)
#
# # RF
# model = RandomForestClassifier(n_estimators=n_estimators)
# model.fit(x_train, y_train)
# joblib.dump(model, 'models/target_models/{}_rf.model'.format(data_name))
# y_pre = model.predict(x_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)
#
# # xgboost
# model = XGBClassifier(n_estimators=n_estimators)
# model.fit(x_train, y_train)
# joblib.dump(model, 'models/target_models/{}_xgboost.model'.format(data_name))
# y_pre = model.predict(x_test)
# acc = accuracy_score(y_pre, y_test)
# print(acc)

# lightgbm
model = LGBMClassifier(n_estimators=n_estimators)
model.fit(x_train, y_train)
joblib.dump(model, 'models/target_models/{}_lgb.model'.format(data_name))
y_pre = model.predict(x_test)
acc = accuracy_score(y_pre, y_test)
print(acc)


