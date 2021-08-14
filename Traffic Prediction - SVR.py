#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.preprocessing import LabelEncoder 


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


#importing the dataset
dataset = pd.read_csv("Dataset.csv")
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
dataset['Date']= le.fit_transform(dataset['Date'])
X = dataset.iloc[:, 1:6].values
y = dataset.iloc[:, 6:7].values


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[14]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[18]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,np.ravel(y_train,order='C'))


# In[19]:


y_pred = regressor.predict(X_test)


# In[20]:


if(y_pred.all()<2.5):
    y_pred=np.round(y_pred-0.5)
    
else:
    y_pred=np.round(y_pred+0.5)


# In[21]:


df1=(y_pred-y_test)/y_test
df1=round(df1.mean()*100,2)
print("Error = ",df1,"%") 
a=100-df1
print("Accuracy= ",a,"%")


# In[22]:


y_test


# In[ ]:





# In[27]:


y_pred


# In[ ]:




