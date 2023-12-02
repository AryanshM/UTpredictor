#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns


# In[5]:


newdata=pd.read_csv(r"C:\Users\Asus\Downloads\UtPredictor\ST2ml.csv", encoding='unicode_escape')


# In[6]:


newdata


# ## Seems like all the odd rows are blank 

# In[7]:


dataseteven=newdata


# In[8]:


dataseteven


# ## we would ideally like to use the rows that have no debarred or absent values for the prediction model

# In[9]:


dataseteven


# In[10]:


dataseteven.isna().sum()


# In[11]:


df_filtered=dataseteven.dropna()


# In[12]:


df_filtered.isna().sum()


# In[13]:


df_filtered


# In[14]:


df_filtered.columns


# In[15]:


columns_to_check = ['CHEMISTRY', 'MATHS', 'ELECTRONICS', 'MECHANICAL', 'SOFTSKILLS',
                    'TOTAL', 'PERCENTAGE', 'CHEMISTRYPUT', 'MATHSPUT',
                    'ELECTRONICSPUT', 'MECHANICALPUT', 'SOFTSKILLSPUT', 'TOTALPUT',
                    'PERCENTAGEPUT']


# In[16]:


df_filtered


# In[17]:


df_filtered.isna().sum()


# In[18]:


df_filtered


# In[19]:


df_filtered.dropna()


# In[20]:


sns.pairplot(df_filtered)


# # Training the Model

# In[21]:


df=df_filtered


# In[22]:


df


# In[23]:


df.columns


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[25]:


STfields = ['CHEMISTRY', 'MATHS', 'ELECTRONICS', 'MECHANICAL', 'SOFTSKILLS', 'TOTAL', 'PERCENTAGE']


# In[26]:


PUTfields = ['CHEMISTRYPUT', 'MATHSPUT', 'ELECTRONICSPUT', 'MECHANICALPUT', 'SOFTSKILLSPUT', 'TOTALPUT', 'PERCENTAGEPUT']


# In[27]:


X = df[STfields]


# In[28]:


y = df[PUTfields]


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=30)


# In[30]:


model = LinearRegression()


# In[31]:


model.fit(X_train, y_train)


# In[32]:


predictions = model.predict(X_test)


# In[33]:


STnew_data = {
    'CHEMISTRY': [25],
    'MATHS': [34],
    'ELECTRONICS': [36],
    'MECHANICAL': [31],
    'SOFTSKILLS': [31],
    'TOTAL': [157],
    'PERCENTAGE': [62.8],
}


# In[34]:


STnew_data_df = pd.DataFrame(STnew_data)
new_predictions = model.predict(STnew_data_df[STfields])
print(f'Predictions for UT:\n{new_predictions}')


# In[ ]:





# In[ ]:





# In[ ]:




