#!/usr/bin/env python
# coding: utf-8

# In[1]:


# packages for data analysis
import numpy as np
import pandas as pd
from sklearn import svm

# visual your data
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


recipes = pd.read_csv('cupcake_muffin.csv')
print(recipes.head())


# In[8]:


# plot our data

sns.lmplot('four', 'sugar', data=recipes, hue='type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70});


# In[12]:


# format or pre process our data

type_label = np.where(recipes['type']=='muffin', 0,1)
recipe_features = recipes.columns.values[1:].tolist()
recipe_features
ingredients = recipes [['four', 'sugar']].values
print(ingredients)


# In[13]:


# fit model
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)


# In[15]:


# get the seporating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30,60)
yy = a * xx - (model.intercept_[0]) / w[1]


# plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# In[20]:


sns.lmplot('four', 'sugar', data=recipes, hue='type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')


# In[25]:


# create function to predict muffin or cupcake
def muffin_or_cupcake(four, sugar):
    if(model.predict([[four, sugar]]))==0:
        print ('You\'re looking at a muffin recipe!')
    else:
        print('You\'re looking at a cupcake recipe!')
        
# predcit if 50 parts flour and 20 parts sugar
muffin_or_cupcake(40, 20)


# In[26]:


# let's plot this on the graph
sns.lmplot('four', 'sugar', data=recipes, hue='type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(40,20,'yo', markersize='9')


# In[ ]:




