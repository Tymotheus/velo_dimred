#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
mpl.rcParams["figure.facecolor"] = "white"

mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white"



# In[2]:



import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


# In[3]:


from calibration_dataset import Tell1Dataset

class MyDS(Tell1Dataset):
	filename_format = '%Y-%m-%d'
	filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'

datapath = "data/calibrations/"
data_list = MyDS.get_filepaths_from_dir(datapath)
mds = MyDS(data_list, read=True)


# In[4]:





# In[5]:


from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


# In[6]:


mds.dfh['R'].df


# In[7]:


dfh = mds.dfh.df.iloc[:,9:]
dfh_r = mds.dfh['R'].df.iloc[:,9:]
dfh_phi = mds.dfh['phi'].df.iloc[:,9:]


# In[8]:


dfp = mds.dfp.df.iloc[:,9:]
dfp_r = mds.dfp['R'].df.iloc[:,9:]
dfp_phi = mds.dfp['phi'].df.iloc[:,9:]


# In[9]:


from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
import neptune
from pytorch_lightning.loggers.neptune import NeptuneLogger
import os


# In[10]:


def make_loader(dataset):
    train, test = train_test_split(dataset, test_size=0.2)
    train_target = torch.tensor(train.values, dtype=torch.float)
    train_data = torch.tensor(train.values, dtype=torch.float)
    test_target = torch.tensor(test.values, dtype=torch.float)
    test_data = torch.tensor(test.values, dtype=torch.float)
    train_tensor = TensorDataset(train_data, train_target) 
    test_tensor = TensorDataset(test_data, test_target) 
    train_loader = DataLoader(dataset = train_tensor)
    test_loader = DataLoader(dataset = test_tensor)
    return train_loader, test_loader


# In[11]:


from datetime import datetime
datetime.now().strftime("%d_%m_%Y_%H_%M_%S")


# In[12]:


PARAMS = {'max_epochs': 1,
          'learning_rate': 0.005,
          'batch_size': 32,
          'gpus' : 1,
            'name' : 'testing calina'
         }

neptune_logger = NeptuneLogger(
    api_key=os.getenv('NEPTUNE_API_TOKEN'),
    project_name="pawel-drabczyk/velodimred",
    experiment_name="small-net testing dfh more epochs",
    params=PARAMS,
    tags=["small-net","dfh","more-epochs"],  
    upload_source_files=['analyze_Pawel.ipynb', 'networks.py']
)

if not os.path.exists('models/{}'.format(PARAMS['name'])):
    os.makedirs('models/{}'.format(PARAMS['name'])) 


# In[13]:


def make_model_trainer(s):
    s = 2048
    dec = VeloDecoder(s)
    enc = VeloEncoder(s)
    model = VeloAutoencoderLt(enc, dec)
    
    tr = pl.Trainer(logger=neptune_logger, max_epochs=PARAMS['max_epochs'], gpus=PARAMS['gpus'])
    return model, tr


# In[14]:


def run_experiment(dataset):
    train_loader, test_loader = make_loader(dataset)
    s = dataset.shape[1]
    model, tr = make_model_trainer(s)
    tr.fit(model, train_loader, test_loader)
    
    tr.save_checkpoint('models/{}/trained_model.ckpt'.format(PARAMS['name']))
    neptune_logger.experiment.log_artifact('models/{}/trained_model.ckpt'.format(PARAMS['name']))
    


# In[15]:


run_experiment(dfh)


# In[16]:


#run_experiment(dfp, "dfp_small")
#run_experiment(dfp_r, "dfp_r_small")
#run_experiment(dfh_r, "dfh_r_small")
#run_experiment(dfp_phi, "dfp_phi_small")
#run_experiment(dfh_phi, "dfh_phi_small")


# In[17]:


#%tensorboard --logdir lightning_logs --host 0.0.0.0


# In[ ]:




