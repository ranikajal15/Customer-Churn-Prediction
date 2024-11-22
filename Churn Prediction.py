#!/usr/bin/env python
# coding: utf-8

# In[28]:


#installing libraries 
get_ipython().system('pip install openpyxl')
get_ipython().system('pip install fancyimpute')
get_ipython().system('pip install missingno')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install scikit-optimize')


# In[29]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from fancyimpute import IterativeImputer
from skopt import BayesSearchCV

import warnings
warnings.filterwarnings('ignore')


# In[30]:


#read file and display data dictionary
description = pd.read_excel('C:/Users/rajpu/Downloads/ECommerceDataset.xlsx', sheet_name='Data Dict', header=1, usecols=[1,2,3])
data = pd.read_excel('C:/Users/rajpu/Downloads/ECommerceDataset.xlsx', sheet_name='E Comm')
description


# In[31]:


#preview the data file
data.head()


# In[32]:


#check the shape of the dataset
print(data.shape)


# In[33]:


#printing summary statistics
data.describe(include='all')


# In[34]:


ax = sns.countplot(x='Churn', data=data, palette=['lightpink', 'skyblue']) 
for a in ax.patches:
    ax.annotate(format((a.get_height()/5630)*100, '.2f'), 
                (a.get_x() + a.get_width()/2., a.get_height()), 
                ha='center', va='center', size=12, xytext=(0, -10), textcoords='offset points')
plt.title("Count of Customers by Churn Status")
plt.show()


# In[35]:


#Distribution of the Tenure of the customers on the platform
sns.displot(x='Tenure', kde=True, data=data, color='purple')
plt.title("Analyzing Customer Retention Periods")
plt.show()


# In[36]:


#Distribution of Order Count of customers
sns.displot(x='OrderCount', kde=True, data=data, color='green')
plt.title("Analyzing Order Counts Per Customer")
plt.show()


# In[37]:


#Distribution of Recency of the customers
sns.displot(x='DaySinceLastOrder', kde=True, data=data, color='red')
plt.title("Analyzing Recency of Customer Engagement")
plt.show()


# In[38]:


#Distribution of Amount returned for money spent by customers
sns.displot(x='CashbackAmount',kde=True, data=data, color='yellow')
plt.title('Customer Cashback Behavior Analysis')
plt.show()


# In[39]:


#Distribution of distance of Warehouse to customers home
sns.displot(x='WarehouseToHome', kde=True, data=data, color='pink')
plt.title("Analysis of Warehouse-to-Customer Distance")
plt.show()


# In[40]:


#Distribution of Percentage increase in customer orders
sns.displot(x='OrderAmountHikeFromlastYear', kde=True, data=data, color='skyblue')
plt.title("Analysis of Customer Order Growth Rates")
plt.show()


# In[41]:


#Distribution of Hours spent on the app by the customers
axx = sns.countplot(x='HourSpendOnApp', data=data)
for a in axx.patches:
    axx.annotate(format((a.get_height()/5630)*100,'.2f'), (a.get_x() + a.get_width()/2., a.get_height()),\
                ha='center',va='center',size=12,xytext=(0, 6),textcoords='offset points')
plt.title("Distribution of hours spent on the app by the customers")
plt.show()


# In[42]:


#Distribution Satisfaction score for churned and retained customers
sns.countplot(x='SatisfactionScore', hue='Churn', palette='magma', data=data)
plt.title("Distribution of Satisfaction Score for Churned and Retained customers")
plt.show()


# In[43]:


#Distribution of Gender for churned and retained customers
sns.countplot(x='Gender', hue='Churn', palette='cividis', data=data)
plt.title("Distribution of Gender for Churned and Retained customers")
plt.show()


# In[44]:


#Distribution of marital status for churned and retained customers
sns.countplot(x='MaritalStatus', hue='Churn', palette='viridis', data=data)
plt.title("Distribution of marital status for churned and retained customers")
plt.show()


# In[45]:


#Distribution of complain for churned and retained customers
sns.countplot(x='Complain', hue='Churn', palette='plasma', data=data)
plt.title("Distribution of complain for churned and retained customers")
plt.show()


# In[46]:


#Relationship between the Tenure and Churn rate
sns.scatterplot(x=data['Tenure'],y=data.groupby('Tenure').Churn.mean())
plt.title("Relationship between Tenure and Churn rate")
plt.show()


# In[48]:


sns.scatterplot(x=data['OrderCount'],y=data.groupby('OrderCount').Churn.mean())
plt.title("Relationship between OrderCount and Churn rate")
plt.show()


# In[49]:


sns.scatterplot(x=data['CouponUsed'],y=data.groupby('CouponUsed').Churn.mean())
plt.title("Relationship between CouponUsed and Churn rate")
plt.show()


# In[54]:


numeric_data = data.drop('CustomerID', axis=1).select_dtypes(include=['number'])

#Correlationmatrix
plt.figure(figsize=(8, 4))
sns.heatmap(numeric_data.corr(), annot=True, cmap='RdYlGn')
plt.title("Correlation Matrix for the Customer Dataset")
plt.show()


# In[55]:


#DATA PREPROCESSING

#count the number of missing values in each column
data.isnull().sum()


# In[57]:


#count the total number of missing values
print(f'Number of missing values in the dataset: {data.isnull().sum().sum()}')
#count the number of rows with missing values
print(f'Number of rows with missing values: {data[data.isnull().any(axis=1)].shape[0]}')


# In[63]:


#plot heatmap of missing values
msno.heatmap(data,figsize=(6,4), cmap='Spectral')
plt.show()


# In[64]:


#Categorical encoding for machine learning models
cat_data = data.select_dtypes(include='object')
cat_data


# In[65]:


#encode categorical variables and add it to the normal dataset
encoded = pd.get_dummies(cat_data,drop_first=True)

data_enc = pd.concat([data.drop(cat_data.columns, axis=1), encoded], axis=1)
data_enc.drop('CustomerID', axis=1,inplace=True)


# In[66]:


#selecting features and target variable, and splitting the data
X=data_enc.drop(['Churn'],axis=1)
y=data_enc['Churn']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

#cross validation
skfcv=StratifiedKFold(n_splits=5)


# In[67]:


#LOGISTICS REGRESSION
log_pipe = Pipeline([('imputer',IterativeImputer(random_state=0)),('scaler',StandardScaler()),
                    ('logreg',LogisticRegression())])

#cross validate logistic regression model
print(f"Cross validation score for Logistic Regression: {np.mean(cross_val_score(log_pipe, X_train, y_train, cv=skfcv, scoring='f1'))}")


# In[70]:


#use logistic regression pipeline to predict test sample
log_pipe.fit(X_train,y_train)
print(f"Test score for Logistic Regression: {f1_score(y_test, log_pipe.predict(X_test))}")
print(f"Training score for Logistic Regression: {f1_score(y_train, log_pipe.predict(X_train))}")
log_mat = confusion_matrix(y_test, log_pipe.predict(X_test))
sns.heatmap(log_mat, annot=True,fmt="g", cmap='coolwarm')
plt.show()


# In[71]:


#RANDOM FOREST
rf_pipe = Pipeline([('imputer',IterativeImputer(random_state=0)),('scaler',StandardScaler()),
                    ('rfmodel',RandomForestClassifier())])

#cross validate Random Forest model
print(f"Cross validation score for Random Forest: {np.mean(cross_val_score(rf_pipe, X_train, y_train, cv=skfcv, scoring='f1'))}")


# In[72]:


#use Random Forest pipeline to predict test and train sample
rf_pipe.fit(X_train,y_train)
print(f"Test score for Random Forest: {f1_score(y_test, rf_pipe.predict(X_test))}")
print(f"Training score for Random Forest: {f1_score(y_train, rf_pipe.predict(X_train))}")
rf_mat = confusion_matrix(y_test, rf_pipe.predict(X_test))
sns.heatmap(rf_mat, annot=True,fmt="g")
plt.show()


# In[73]:


#XGBOOST
xgb_pipe = Pipeline([('imputer',IterativeImputer(random_state=0)),('scaler',StandardScaler()),
                    ('xgb',XGBClassifier(verbosity=0,use_label_encoder=False))])

#cross validate XGBoost model
print(f"Cross validation score for XGBoost: {np.mean(cross_val_score(xgb_pipe, X_train, y_train, cv=skfcv, scoring='f1'))}")


# In[74]:


#use XGBoost pipeline to predict test and train sample
xgb_pipe.fit(X_train,y_train)
print(f"Test score for XGBoost: {f1_score(y_test, xgb_pipe.predict(X_test))}")
print(f"Training score for XGBoost: {f1_score(y_train, xgb_pipe.predict(X_train))}")
xgb_mat = confusion_matrix(y_test, xgb_pipe.predict(X_test))
sns.heatmap(xgb_mat, annot=True, fmt="g", cmap='PiYG')
plt.show()


# In[75]:


#HYPERPARAMETER TUNING
#create hyperparameter search space
space={'xgb__eta': (0.01,0.3),
       'xgb__max_depth': (5,11),
      'xgb__subsample': (0.4,1),
      'xgb__n_estimators': (100,250),
      'xgb__gamma':(0,5),
      'xgb__colsample_bytree':(0.4,1),
      'xgb__min_child_weight': (0.3,1)}


# In[76]:


#create BayesSearchCV object
search=BayesSearchCV(xgb_pipe,search_spaces=space,n_jobs=-1,cv=skfcv, scoring='f1')


# In[77]:


#fit the object to the data
search.fit(X_train, y_train)


# In[78]:


#print the best cv score and the best parameters
print(search.best_score_)
print(search.best_params_)


# In[79]:


#use the best model parameters to predict the test sample and print the results
best_model=search.best_estimator_
print(f1_score(y_test,best_model.predict(X_test)))
sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)), annot=True, fmt="g")
plt.show()


# In[80]:


feature_importance=pd.DataFrame(search.best_estimator_[2].feature_importances_, columns=['importance'])
feature_importance['features'] = X_train.columns

plt.figure(figsize=(10,8))
sns.barplot(x='importance', y='features', data=feature_importance.sort_values(by='importance', ascending=False))
plt.title('Feature importances')
plt.show()

