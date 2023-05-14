from xgboost import XGBClassifier
import joblib
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path_data = '/Users/yinghua.li/Documents/Pycharm/MLPrior/data/adult.csv'
label_name = 'income'

model = joblib.load('/Users/yinghua.li/Documents/Pycharm/MLPrior/models/target_models/adult_svm.model')
print(model)

x, y = read_data(path_data, label_name)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)



print(model.support_)

