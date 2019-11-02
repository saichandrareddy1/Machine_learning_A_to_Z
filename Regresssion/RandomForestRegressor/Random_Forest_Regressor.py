#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[2]:


data = pd.read_csv('data.csv')
print(data.head())


# In[3]:


data.rename(columns={'Total Trade Quantity':'Total','Turnover (Lacs)':'Turnover'},inplace='True')
close = data['Close']
data = data.drop(['Close'],axis=1)
data['Close'] = close
print(data.head())


# In[4]:


X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[5]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05,random_state=0)


# In[6]:


regressor = RandomForestRegressor(criterion='mse',random_state=0)
regressor.fit(X_train,y_train)


# In[7]:


y_pred = regressor.predict(X_test)


# In[8]:


score = r2_score(y_pred,y_test)
print(score)


# In[9]:


y_predict = regressor.predict([[230,237,225,226,1708590,3960]])


# In[10]:


y_predict


# In[ ]:




