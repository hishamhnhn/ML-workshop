#!/usr/bin/env python
# coding: utf-8

# In[45]:


import streamlit as st
import seaborn as sns
import numpy as np 
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[65]:


st.title('Intractive plotting')
st.header('Department of Technology')
st.header('The university of Lahore')
st.header('Developed by Hisham')


# In[47]:


#st.image('ai.png')


# In[ ]:





# In[48]:


dataset_name = st.sidebar.selectbox(
    "select dataset",
    ("iris", "wine",'breast_cancer')
)


# In[49]:


classifier_name = st.sidebar.selectbox(
    "select model",
    ("knn", "random forest", "SVM")
)


# In[50]:


def get_dataset(dataset_name):
    data = None
    if dataset_name == 'iris':
        data = datasets.load_iris()
    elif dataset_name == 'wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y


# In[51]:


#dataset_name = ['irsi','wine','canser']
x,y=get_dataset(dataset_name) # assigning to the return variable x and y while 
                  # while geting the dataset


# In[52]:


st.header('Datasets')


# In[53]:


#st.write(get_dataset(dataset_name))


# In[54]:


st.write('Shape of dataset:', x.shape)


# In[55]:


st.write('number of classes:', len(np.unique(y))) # getting the number of classes in the dataset 


# In[56]:


def add_parameter_ui(classifier_name):
    params = dict() # creat an empty dictionary 
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C',0.01,10.0)
        params['C'] = C  # C is degree of correct classification 
    elif classifier_name == 'knn':  
        k = st.sidebar.slider('k',1,15)
        params['k'] = k   # k is the nearest neighbours 
    else:
        max_depth = st.sidebar.slider('max_depth',2,15) # max depth is the density of forest 
        params['max_depth'] = max_depth 
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators  # n is number of trees 
    return params
        


# In[57]:


params = add_parameter_ui(classifier_name)


# In[58]:


def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'knn':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                          max_depth=params['max_depth'],random_state=1234)
    return clf


# In[59]:


clf = get_classifier(classifier_name,params)


# In[60]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[61]:


clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[62]:


acc = accuracy_score(y_test,y_pred)
st.write(f'Classifies = {classifier_name}')
st.write(f'Accuracy =',acc)


# In[63]:


# for plotting using PCA which is used for the reduction of the feature 

pca = PCA(2)
x_projected = pca.fit_transform(x)


# In[64]:


x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,
            c=y,alpha=0.8,
           cmap='viridis')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
st.pyplot(fig)



#fig = px.scatter(x1,x2,animation_frame=x1,animation_group=x2)
#st.write(fig)


# In[41]:


#st.write(x1),st.write(x2)


# In[ ]:




