from xgboost import XGBClassifier
import joblib
from utils import *
from sklearn.model_selection import train_test_split

path_data = '/Users/yinghua.li/Documents/Pycharm/MLPrior/data/adult.csv'
label_name = 'income'

model = joblib.load('/Users/yinghua.li/Documents/Pycharm/MLPrior/models/target_models/adult_xgboost.model')
print(model)

x, y = read_data(path_data, label_name)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(XGBClassifier.get_xgb_params())
print(XGBClassifier.get_booster())


# XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
#               colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
#               early_stopping_rounds=None, enable_categorical=False,
#               eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
#               grow_policy='depthwise', importance_type=None,
#               interaction_constraints='', learning_rate=0.300000012,
#               max_bin=256, max_cat_threshold=64, max_cat_to_onehot=4,
#               max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
#               missing=nan, monotone_constraints='()', n_estimators=30, n_jobs=0,
#               num_parallel_tree=1, predictor='auto', random_state=0, ...)