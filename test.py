from xgboost import XGBClassifier
import joblib
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('./data/adult.csv')
print(df.head(3))
print(df['hours-per-week'].dtypes)

df['hours-per-week'][0]=0
print(df.head(3))
print(df['hours-per-week'].dtypes)