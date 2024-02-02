#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('diabetes1.csv')


# In[3]:


data.head()


# In[4]:


data


# In[5]:


data.tail()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[9]:


data.isnull().sum()


# In[10]:


sns.countplot(x='Pregnancies',data=data)


# In[14]:


data.Pregnancies.value_counts()


# In[15]:


#multiple lines
plt.figure(figsize=(20,25),facecolor='white')
plotnumber=1

for column in data:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.histplot(data[column])
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Count',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[16]:


sns.countplot(x='Pregnancies',hue='Outcome',data=data)
plt.show()


# In[17]:


sns.histplot(x='Glucose',hue='Outcome',data=data)


# In[18]:


sns.relplot(x='Glucose',y='BloodPressure',hue='Outcome',data=data)
plt.show()


# In[19]:


sns.relplot(x='Glucose',y='SkinThickness',hue='Outcome',data=data)
plt.show()


# In[20]:


sns.histplot(x='BloodPressure',hue='Outcome',data=data)


# In[21]:


sns.relplot(x='BloodPressure',y='SkinThickness',hue='Outcome',data=data)
plt.show()


# In[22]:


#anaylze bp with insulin
sns.relplot(x='BloodPressure',y='Insulin',col='Outcome',data=data)
plt.show()


# In[23]:


#anaylze with insulin with target
sns.histplot(x='Insulin',hue='Outcome',data=data)


# In[24]:


#handling corrupted data
data.Glucose.replace(0,np.median(data.Glucose),inplace=True)
data.loc[data['Glucose']==0]


# In[25]:


data.loc[data['BMI']==0]


# In[26]:


#chcek outlier
data.columns


# In[27]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
d1=['Pregnancies','Outcome']
data1=sc.fit_transform(data.drop(d1,axis=1))


# In[28]:


con_data=data[['Pregnancies','Outcome']]


# In[29]:


d1


# In[31]:


con_data


# In[32]:


type(data1)
data2=pd.DataFrame(data1,columns=['Gucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])


# In[33]:


final_df=pd.concat([data2,con_data],axis=1)


# In[35]:


final_df


# In[36]:


sns.heatmap(data2.corr(),annot=True)


# In[37]:


#model creation
X=final_df.iloc[:,:-1]
Y=final_df.Outcome


# In[38]:


Y


# In[39]:


#train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=45)


# In[40]:


Y_test


# In[41]:


from sklearn.linear_model  import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,Y_train)


# In[42]:


Y_pred=clf.predict(X_test)


# In[43]:


Y_pred


# In[44]:


Y_pred_prob=clf.predict_proba(X_test)


# In[45]:


Y_pred_prob


# In[ ]:




