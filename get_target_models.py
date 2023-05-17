import joblib
import argparse
from get_rank_idx import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


ap = argparse.ArgumentParser()
ap.add_argument("--path_data", type=str)
ap.add_argument("--label_name", type=str)
ap.add_argument("--n_estimators", type=int)
args = ap.parse_args()

# python get_target_models.py --path_data 'data/adult.csv' --label_name 'income'
# python get_target_models.py --path_data 'data/bank.csv' --label_name 'y'
# python get_target_models.py --path_data 'data/stroke.csv' --label_name 'stroke'


path_data = args.path_data
label_name = args.label_name
n_estimators = 5


# path_data = 'data/heart.csv'
# label_name = 'label'
# n_estimators = 5


def main():

    data_name = path_data.split('/')[-1].split('.')[0]
    x, y = read_data(path_data, label_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # adult LR
    model = LogisticRegression()
    model.fit(x_train, y_train)
    joblib.dump(model, 'models/target_models/{}_lr.model'.format(data_name))
    y_pre = model.predict(x_test)
    acc = accuracy_score(y_pre, y_test)
    print('test', acc)


    # tree
    model = DecisionTreeClassifier(min_samples_leaf=10)
    model.fit(x_train, y_train)
    joblib.dump(model, 'models/target_models/{}_dtree.model'.format(data_name))
    y_pre = model.predict(x_test)
    acc = accuracy_score(y_pre, y_test)
    print('test', acc)

    # xgboost
    model = XGBClassifier(n_estimators=n_estimators)
    model.fit(x_train, y_train)
    joblib.dump(model, 'models/target_models/{}_xgb.model'.format(data_name))
    y_pre = model.predict(x_test)
    acc = accuracy_score(y_pre, y_test)
    print('test', acc)
    # NB
    model = GaussianNB()
    model.fit(x_train, y_train)
    joblib.dump(model, 'models/target_models/{}_nb.model'.format(data_name))
    y_pre = model.predict(x_test)
    acc = accuracy_score(y_pre, y_test)
    print('test', acc)

    # KNN
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, 'models/target_models/{}_knn.model'.format(data_name))
    y_pre = model.predict(x_test)
    acc = accuracy_score(y_pre, y_test)
    print('test', acc)


if __name__ == '__main__':
    main()

