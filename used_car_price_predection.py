#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[63]:


df_train = pd.read_csv('car_train-data.csv')
df_test = pd.read_csv('car_test-data.csv')


# In[64]:


df_train.head()


# In[65]:


df_train.info()


# In[ ]:





# In[66]:


df_train.shape


# In[67]:


df_train.isnull().sum()


# In[68]:


df_train = df_train.rename(columns = {'Unnamed: 0': 'id'})


# In[73]:


df_train["Seats"].fillna(value = 5.0, inplace=True)
df_train.Seats[df_train.Seats == 0.0] = 5.0
df_train.isna().sum()

df_train.Mileage[df_train.Mileage == '0.0 kmpl'] = np.nan
df_train['Mileage'] = df_train['Mileage'].apply(lambda x: re.sub(r'(\d+\.\d+)\s(kmpl|km\/kg)', 
                                                                 r'\1', str(x)))
df_train['Mileage'] = df_train['Mileage'].astype(float)
df_train['Mileage'].mode()
df_train['Mileage'].fillna(value = 17.0, inplace = True)
df_train.isna().sum()

df_train['Engine'] = df_train['Engine'].apply(lambda x: re.sub(r'(\d+)\s(CC)', r'\1', str(x)))
df_train['Engine'] = df_train['Engine'].astype(float)
df_train['Engine'].mode()
df_train['Engine'].fillna(value = 1197.0, inplace = True)
df_train.isna().sum()

df_train['Power'] = df_train['Power'].str.split(' ').str[0]
df_train.Power[df_train.Power == 'null'] = np.NaN
df_train['Power'].isnull().sum()
df_train['Power'].fillna(value = 74, inplace = True)
df_train.isna().sum()


# In[74]:


df_train.isna().sum()


# In[75]:


df_train['Name'] = df_train['Name'].str.split(' ').str[0]
df_train.groupby('Name')['id'].nunique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


df_train.Name[df_train.Name == 'Isuzu'] = 'ISUZU'


# In[77]:


del df_train['New_Price']


# In[78]:


dataset = df_train.copy()


# In[79]:


del df_train['id']


# In[80]:


df_train.dtypes


# In[81]:


df_train['Year'] = df_train['Year'].astype(float)
df_train['Kilometers_Driven'] = df_train['Kilometers_Driven'].astype(float)


# In[82]:


df_train['Price_log'] = np.log1p(df_train['Price'].values)
del df_train['Price']


# In[83]:


df_train = pd.get_dummies(df_train, drop_first = True)


# In[84]:


X = df_train.drop(columns = ['Price_log'], axis = 1)
y = df_train.iloc[:, 6].values


# In[85]:


from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import r2_score
import re
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
get_ipython().run_line_magic('matplotlib', 'inline')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[86]:


regressor_1 = LinearRegression()
regressor_1.fit(X_train, y_train)


# In[87]:


y_pred_1 = regressor_1.predict(X_test)


# In[88]:


regressor_1.score(X_test,y_test)


# In[89]:


regressor_2 = RandomForestRegressor(random_state = 0)
regressor_2.fit(X_train, y_train)


# In[90]:


y_pred_2 = regressor_2.predict(X_test)


# In[91]:


regressor_2.score(X_test,y_test)


# In[92]:


regressor_3 = DecisionTreeRegressor(random_state = 0)
regressor_3.fit(X_train, y_train)


# In[93]:


y_pred_3 = regressor_3.predict(X_test)


# In[94]:


regressor_3.score(X_test, y_test)


# In[95]:


plt.style.use('ggplot')
colors = ['#FF8C73','#66b3ff','#99ff99','#CA8BCA', '#FFB973', '#89DF38', '#8BA4CA', '#ffcc99', 
          '#72A047', '#3052AF', '#FFC4C4']


# In[96]:


plt.figure(figsize = (10,8))
bar1 = sns.countplot(dataset['Year'])
bar1.set_xticklabels(bar1.get_xticklabels(), rotation = 90, ha = 'right')
plt.title('Count year wise', size = 24)
plt.xlabel('Year', size = 18)
plt.ylabel('Count', size = 18)
plt.show()


# In[97]:


plt.figure(figsize = (5,5))
sns.countplot(dataset['Fuel_Type'])
plt.title('Types of Fuel and count', size = 24)
plt.tight_layout()
plt.show()


# In[98]:


plt.figure(figsize = (6,6))
plt.pie(dataset['Location'].value_counts(), startangle = 90, autopct = '%1.1f%%', colors = colors, 
        labels = dataset['Location'].unique())
centre_circle = plt.Circle((0,0),0.80,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.show()


# In[99]:


plt.figure(figsize = (5,5))
sns.countplot(dataset['Transmission'])
plt.title('Types of transmission', size = 24)
plt.tight_layout()
plt.show()


# In[100]:




