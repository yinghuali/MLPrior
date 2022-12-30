import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/diabetes_data.csv')
print(len(df))

for col in df.columns:
    if df[col].dtypes == 'object':
        df[col] = df[col].factorize()[0]

x = df.to_numpy()[:, 0:-1]
y = df.to_numpy()[:, -1]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)
print('==train=')
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pre = model.predict(x_test)
acc = accuracy_score(y_pre, y_test)
print(acc)


