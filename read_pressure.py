#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('pressure.zip')

print(df.mean(axis=0))

print(df.std(axis=0))

