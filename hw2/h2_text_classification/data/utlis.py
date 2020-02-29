from sklearn.model_selection import train_test_split
import pandas as pd
import numpy

df = pd.read_csv("train.csv", header=0)
y = df.pop('label').to_frame()
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

df1 = X_train.join(y_train)
df2 = X_test.join(y_test)

df1.to_csv("ej_train.csv",index=False)
df2.to_csv("ej_test.csv",index=False)
