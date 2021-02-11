#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
customer = pd.read_table(r'C:\Users\mer00\Downloads\Telco-Customer-Churn.csv',sep=',')
customer.head()


# In[3]:


cols=['SeniorCitizen','gender','tenure','PhoneService','InternetService','StreamingMovies','Churn']
customer[cols]


# In[4]:


customer.dtypes


# In[5]:


customer.describe()


# In[4]:


customer.shape


# In[6]:


cols=['SeniorCitizen','gender','tenure','PhoneService','InternetService','StreamingMovies','Churn']
customer[cols].sort_values(by='tenure', ascending=False)


# In[7]:


customer[customer.tenure<5][cols]


# PART B
# 
# 1.	Describe the customer profile of customer 6467-CHFZW (Tenure, gender, services they have with the company, contract type, if they are signed up for paperless billing, their payment method and total charges)

# In[8]:


customer[customer.customerID=="6467-CHFZW"]


# 2.What is the average monthly charge for customers by gender?
# 3.Create appropriate visualizations to summarize the tenure of customers by age and services they have with the company
# 
# Ans: USED TABLEAU
# 
# 

# In[9]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file
import matplotlib.pyplot as plt


# In[10]:


customer.TotalCharges = pd.to_numeric(customer.TotalCharges, errors='coerce')
customer.MonthlyCharges = pd.to_numeric(customer.MonthlyCharges, errors='coerce')
customer.isnull().sum()


# In[11]:


customer.dropna(inplace = True)
customer.isnull().sum()


# In[12]:


df2 = customer.iloc[:,1:]
df2.head()


# 2.	Prepare features for your model. Please identify what methods were used for which features.

# In[13]:


#Binary encoding
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)
df2.head()


# In[14]:


#One Hot Encodin
df_dummies = pd.get_dummies(df2)
df_dummies.dtypes


# In[17]:


df_dummies.head()


# 1.Identify the target(s) and features that you would use for the task of predicting customer churn using the data source. Comment on how well the features are related to the target and provide any visualization/summary statistics

# In[16]:


plt.figure(figsize=(15,8))
df_dummies.corr()['Churn'].sort_values(ascending = True).plot(kind='bar',color=['Green'])


# In[17]:


from sklearn.metrics import average_precision_score


# In[18]:


#logistic regression
y = df_dummies['Churn'].values
X = df_dummies.drop(columns = ['Churn'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
from sklearn.linear_model import LogisticRegression
model =  LogisticRegression(solver='liblinear',class_weight=None, max_iter=100)

result = model.fit(X_train, y_train)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))
#print(average_precision_score(y_test, prediction_test))


# In[19]:


#KNN Algorithm
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,X,y,cv=10,scoring="accuracy")
print(scores.mean())
#scoresp=cross_val_score(knn,X,y,cv=10,scoring="precision")
#print(scoresp.mean())


# In[20]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))
#print (average_precision_score(y_test, prediction_test))


# In[23]:


#XGBOOST
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)


# In[24]:


#decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
scr=cross_val_score(clf, X_train, y_train, cv=10)
print(scr.mean())

