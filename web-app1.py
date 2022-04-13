#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[23]:


st.title('Intractive plotting')
st.header('Department of Technology')
st.header('The university of Lahore')


# In[3]:


#st.image('ai.png')


# In[25]:


import base64

st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data1 = f.read()
    return base64.b64encode(data1).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('ai.png')


# In[5]:


dataset_name = st.sidebar.selectbox(
    "select dataset",
    ("iris", "wine",'breast_cancer')
)


# In[6]:


classifier_name = st.sidebar.selectbox(
    "select model",
    ("knn", "random forest", "SVM")
)


# In[7]:


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


# In[8]:


#dataset_name = ['irsi','wine','canser']
x,y=get_dataset(dataset_name) # assigning to the return variable x and y while 
                  # while geting the dataset


# In[9]:


st.header('Datasets')


# In[10]:


#st.write(get_dataset(dataset_name))


# In[11]:


st.write('Shape of dataset:', x.shape)


# In[12]:


st.write('number of classes:', len(np.unique(y))) # getting the number of classes in the dataset 


# In[13]:


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
        


# In[14]:


params = add_parameter_ui(classifier_name)


# In[15]:


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


# In[16]:


clf = get_classifier(classifier_name,params)


# In[17]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[18]:


clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[19]:


acc = accuracy_score(y_test,y_pred)
st.write(f'Classifies = {classifier_name}')
st.write(f'Accuracy =',acc)


# In[20]:


# for plotting using PCA which is used for the reduction of the feature 

pca = PCA(2)
x_projected = pca.fit_transform(x)


# In[44]:


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




