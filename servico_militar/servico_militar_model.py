#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

class servico_militar_model():
    
    def __init__(self, model_file, scaler_file):
        # read the 'model' and 'scaler' files wich were saved
        with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
    
    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file,delimiter=',')
        self.df_with_predictions = df.copy()
        
        df['PESO'] = df['PESO'].fillna(round(df['PESO'].mean(),2))
        df['ALTURA'] = df['ALTURA'].fillna(round(df['ALTURA'].mean(),2))
        df['CABECA'] = df['CABECA'].fillna(round(df['CABECA'].mean(),2))
        df['CALCADO'] = df['CALCADO'].fillna(round(df['CALCADO'].mean(),2))


# In[ ]:




