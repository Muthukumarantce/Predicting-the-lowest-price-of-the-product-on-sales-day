#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 23:52:29 2020

@author: Muthu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import xgboost as xgb
import lightgbm

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
test_data = pd.read_csv("data/test.csv")
train_data = pd.read_csv("data/train.csv")
print(test_data.head(5))
print(train_data.head(5))


#Check number of Categorical and numberical variables
train_data.select_dtypes(include=['int64','float64']).columns
train_data.select_dtypes(include=['object']).columns


#Date Feature Engineering
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data['year'] = pd.DatetimeIndex(train_data['Date']).year
train_data['month'] = pd.DatetimeIndex(train_data['Date']).month
train_data['day'] = pd.DatetimeIndex(train_data['Date']).day
train_data['dayofweek'] = pd.DatetimeIndex(train_data['Date']).dayofweek

print(train_data.columns)

#Univariate Analysis
plt.hist(train_data['State_of_Country'])
plt.hist(train_data['Market_Category'])
plt.hist(train_data['Product_Category'])
plt.hist(train_data['Grade'])
plt.hist(train_data['Demand'])
plt.hist(train_data['Low_Cap_Price'])
plt.hist(train_data['High_Cap_Price'])




#Missing features
missing_features = train_data.columns[train_data.isnull().any()]
missing_features



#Train Test Data Splitup
X=train_data.drop(['Low_Cap_Price','Item_Id','Date'],axis=1)
y=train_data['Low_Cap_Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=20,test_size=0.2)


print(X_train.columns)

num_cols = ['State_of_Country', 'Market_Category', 'Product_Category', 'Grade',
       'Demand', 'High_Cap_Price', 'year', 'month', 'day', 'dayofweek']

X_train_stand = X_train.copy()
X_test_stand = X_test.copy()



for i in num_cols:
    
    # fit on training data column
    scale = StandardScaler().fit(X_train_stand[[i]])
    
    # transform the training data column
    X_train_stand[i] = scale.transform(X_train_stand[[i]])
    
    # transform the testing data column
    X_test_stand[i] = scale.transform(X_test_stand[[i]])


X_train_stand.describe()

##Apply MinMaxScaling to numerical features

X_train_norm = X_train.copy()
X_test_norm = X_test.copy()

for i in num_cols:
    
    # fit on training data column
    norm = MinMaxScaler().fit(X_train_norm[[i]])
    
    # transform the training data column
    X_train_norm[i] = norm.transform(X_train_norm[[i]])
    
    # transform the testing data column
    X_test_norm[i] = norm.transform(X_test_norm[[i]])
    

X_train_stand.describe()

trainX = [X_train, X_train_norm, X_train_stand]
testX = [X_test, X_test_norm, X_test_stand]

trainX


#Function for Model fitting and metrics evaluation
def runmodel(model,train_data,test_data):
     rmae=[]
     for i in range(len(train_data)):
         #fit
         model.fit(train_data[i],y_train)
         
         #predict
         pred=np.round(model.predict(test_data[i]))
         pred[pred<0] = 0
         
         #RMSE
         rmae.append(max(0,(100-mean_squared_log_error(y_test,pred))))
         
         return rmae
     


         
#Building the model
#Linear Models
#Linear Regression    
lin = LinearRegression() 
rmae_lin =runmodel(lin,trainX,testX)
df_lin = pd.DataFrame({"RMAE":rmae_lin},index=['Original','Normalized','Standardized'])





#Ridge Regression
ridge=Ridge(max_iter=50,alpha=5.0)
rmae_ridge = runmodel(ridge,trainX,testX)
df_ridge = pd.DataFrame({"RMAE":rmae_ridge},index=['Original','Normalized','Standardized'])


#LASSO Regression
#max_iter=300 91.51447
#275 91.51447
#2,200 91.51466
#1.18,200 91.5151
#1.18,50 91.51514
lasso = Lasso(alpha=5.6,max_iter=20,tol=0.0001) 
rmae_lasso = runmodel(lasso,trainX,testX)
df_lasso = pd.DataFrame({'RMAE':rmae_lasso},index=['Original','Normalized','Standardized'])


#XGB
#Model_score = 99.80
model_xgb = xgb.XGBRegressor(objective = 'reg:squarederror',n_estimators=15000, max_depth=1, learning_rate=0.03,gamma=0.22,min_child_weight=4.44,colsample_bytree=1) 
#model_xgb = xgb.XGBRegressor(objective = 'reg:squarederror',n_estimators=700, max_depth=1, learning_rate=0.08,gamma=0.13,min_child_weight=8.88,colsample_bytree=1) 
rmae_xgb= runmodel(model_xgb,trainX,testX)
df_xgb = pd.DataFrame({'RMAE':rmae_xgb},index=['Original','Normalized','Standardized'])


#LGM
#Model_score = 99.84
lgm =lightgbm.LGBMRegressor(silent=True,n_estimators=375,learning_rate=0.03)
#Model_score = 99.83
lgm =lightgbm.LGBMRegressor(silent=True,n_estimators=2166,learning_rate=0.01)
rmae_lgm= runmodel(lgm,trainX,testX)
df_lgm = pd.DataFrame({'RMAE':rmae_lgm},index=['Original','Normalized','Standardized'])




#Hyper parameter tuning for XGB
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
max_depth = [int(x) for x in np.linspace(1,20, num=10)]
gamma = [x for x in np.linspace(0, 0.4, num=10)]
learning_rate = [float(x) for x in np.linspace(0.005, 0.1, num=10)]
min_child_weight = [float(x) for x in np.linspace(0, 10, num=10)]
colsample_bytree = [0.3, 0.5, 0.7, 1]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'learning_rate':learning_rate,
               'gamma':gamma,
               'min_child_weight':min_child_weight,
               'colsample_bytree':colsample_bytree}
model_xgb = xgb.XGBRegressor(random_state=1, objective='reg:squarederror', no_omp=1) #singlethread

xgb_random = RandomizedSearchCV(estimator=model_xgb,#swap in with whatever model you're using
                                param_distributions=random_grid,
                                scoring='neg_mean_absolute_error',
                                n_iter=10,
                                cv=4,
                                n_jobs=-1,
                                verbose=10
                                )

result=xgb_random.fit(X_train_stand, y_train)
params = result.best_params_
params = result.best_score_

result.best_params_

#HyperParameter Tuning for Lasso & Ridge
max_iter = [int(x) for x in np.linspace(start =20,stop=2000,num=10)]
alpha = [x for x in np.linspace(0.1,10,num=10)]
tol=[x for x in np.linspace(0.0001,5,num=10)]
param_grid = dict(max_iter=max_iter,alpha=alpha,tol=tol)
grid = GridSearchCV(estimator=lasso,param_grid = param_grid,cv=4,n_jobs=-1)
grid_result = grid.fit(X_train_stand,y_train)
print(grid_result.best_params_)
print(grid_result.best_score_)


#HyperParameter Tuning for LGBM
n=[int(x) for x in np.linspace(start =500,stop=3000,num=10)]
learning_rate=[0.1,0.2,0.3,0.4,0.5]
param={'n_estimators':n,'learning_rate':learning_rate}
grid = GridSearchCV(estimator=lgm,param_grid = param,cv=4,n_jobs=-1)
grid_result = grid.fit(X_train_stand,y_train)
print(grid_result.best_params_)
print(grid_result.best_score_)

#Test Data
#Date Feature Engineering
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data['year'] = pd.DatetimeIndex(test_data['Date']).year
test_data['month'] = pd.DatetimeIndex(test_data['Date']).month
test_data['day'] = pd.DatetimeIndex(test_data['Date']).day
test_data['dayofweek'] = pd.DatetimeIndex(test_data['Date']).dayofweek



test_x=test_data.drop(['Item_Id','Date'],axis=1)


num_cols = ['State_of_Country', 'Market_Category', 'Product_Category', 'Grade',
       'Demand', 'High_Cap_Price', 'year', 'month', 'day', 'dayofweek']


test_x_stand = test_x.copy()


for i in num_cols:
    
    # fit on training data column
    scale = StandardScaler().fit(test_x_stand[[i]])
    
    # transform the training data column
    test_x_stand[i] = scale.transform(test_x_stand[[i]])
    
    # transform the testing data column
    test_x_stand[i] = scale.transform(test_x_stand[[i]])


test_x_stand.describe()

##Apply MinMaxScaling to numerical features


test_x_norm = test_x.copy()

for i in num_cols:
    
    # fit on training data column
    norm = MinMaxScaler().fit(test_x_norm[[i]])
    
    # transform the training data column
    test_x_norm[i] = norm.transform(test_x_norm[[i]])
    
    # transform the testing data column
    test_x_norm[i] = norm.transform(test_x_norm[[i]])
    

#Test Data Prediction
pred_test_x_lin = lin.predict(test_x_norm)
pred_test_x_ridge = ridge.predict(test_x_norm)
pred_test_x_lasso = lasso.predict(test_x)
pred_test_x_xgb = model_xgb.predict(test_x)
pred_test_x_lgm = lgm.predict(test_x)

pred_test_x_lin = np.round(pred_test_x_lin)
pred_test_x_ridge = np.round(pred_test_x_ridge)
pred_test_x_lasso = np.round(pred_test_x_lasso)
pred_test_x_xgb = np.round(pred_test_x_xgb)
pred_test_x_lgm = np.round(pred_test_x_lgm )


pred_test_x_lin[pred_test_x_lin < 0]=0
pred_test_x_xgb[pred_test_x_xgb < 0]=0
pred_test_x_ridge[pred_test_x_ridge < 0]=0
pred_test_x_lasso[pred_test_x_lasso<0] = 0
pred_test_x_lgm[pred_test_x_lgm<0] = 0

final = 0.8*pred_test_x_lgm + 0.2*pred_test_x_xgb
#final=pred_test_x_lgm

submission_file = pd.DataFrame({'Item_Id':test_data['Item_Id'],'Low_Cap_Price':final})

submission_file.to_csv("submission_file.csv",index=False)

