import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV , train_test_split

#Import dataset
df=pd.read_csv('/Users/janaaloud/Desktop/insurance.csv')

#Drop target feature
X = df.drop(["charges"] , axis =1)
y = df.charges
X_train , X_test , y_train , y_test = train_test_split(X,y,random_state = 100 , test_size = 0.3)

#Converting Categorical Features into numerical using LabelEncoder()
le = LabelEncoder()
#For train dataset
X_train["sex"]=le.fit_transform(X_train["sex"])
print(X_train["sex"])
X_train["smoker"]=le.fit_transform(X_train["smoker"])
X_train["region"] = le.fit_transform(X_train["region"])
print(X_train.head())

#For test dataset
X_test["sex"]=le.fit_transform(X_test["sex"])
X_test["smoker"]=le.fit_transform(X_test["smoker"])
X_test["region"] = le.fit_transform(X_test["region"])
print(X_test.head())

##create model
lr = LinearRegression()
rfr = RandomForestRegressor()
dt = DecisionTreeRegressor()
print(lr.fit(X_train,y_train))
print(r2_score(lr.predict(X_train) , y_train))
prediction=lr.predict(X_test)


#Moving it into pickle file to be used in streamlit
import pickle
file = open('/Users/janaaloud/Desktop/logisticmodel.pkl', "wb")
pickle.dump(lr , file)
file.close()
model = open('/Users/janaaloud/Desktop/logisticmodel.pkl', "rb")
forest = pickle.load(model)
