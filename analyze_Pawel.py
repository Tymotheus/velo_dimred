#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from calibration_dataset import Tell1Dataset
class MyDS(Tell1Dataset):
	filename_format = '%Y-%m-%d'
	filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'
datapath = "data/calibrations/"
data_list = MyDS.get_filepaths_from_dir(datapath)
mds = MyDS(data_list, read=True)

from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

dfh = mds.dfh.df.iloc[:,9:]
dfh_r = mds.dfh['R'].df.iloc[:,9:]
dfh_phi = mds.dfh['phi'].df.iloc[:,9:]
dfp = mds.dfp.df.iloc[:,9:]
dfp_r = mds.dfp['R'].df.iloc[:,9:]
dfp_phi = mds.dfp['phi'].df.iloc[:,9:]

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
import neptune
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import os
from datetime import datetime
datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

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

PARAMS = {'max_epochs': 10,
          'learning_rate': 0.0005,
          'batch_size': 32,
          'gpus' : 1,
          # 'name' : 'name',
          'experiment_name' : 'small-net ReLu ReduceLROnPlateau ',
          'tags' : ["small-net","ReLu","ReduceLROnPlateau"],
          'source_files' : ['analyze_Pawel.ipynb', 'networks.py']  
         }



datasetNames = ['dfh', 'dfhr', 'dfhphi', 'dfp', 'dfpr', 'dfpphi']

for d in datasetNames:
    '''loggers[d] = NeptuneLogger(
        api_key=os.getenv('NEPTUNE_API_TOKEN'),
        project_name="pawel-drabczyk/velodimred",
        experiment_name="small-net ReLu {}".format(d),
        params=PARAMS,
        tags=["small-net","ReLu",d],  
        upload_source_files=['analyze_Pawel.ipynb', 'networks.py']
    )'''
    if not os.path.exists('models/{}/{}'.format(PARAMS['experiment_name'], d)):
        os.makedirs('models/{}/{}'.format(PARAMS['experiment_name'], d)) 

def make_model_trainer(s, neptune_logger, lr):
    s = 2048
    dec = VeloDecoder(s)
    enc = VeloEncoder(s)
    model = VeloAutoencoderLt(enc, dec, lr)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tr = pl.Trainer(logger=neptune_logger, callbacks=[lr_monitor],  max_epochs=PARAMS['max_epochs'], gpus=PARAMS['gpus'])
    return model, tr

def run_experiment(dataset, datasetName, par):
    train_loader, test_loader = make_loader(dataset)
    s = dataset.shape[1]
    neptune_logger = NeptuneLogger(
        api_key=os.getenv('NEPTUNE_API_TOKEN'),
        project_name="pawel-drabczyk/velodimred",
        experiment_name=par['experiment_name'],
        params=par,
        tags=par['tags'] + [datasetName],  
        upload_source_files= par['source_files']
    )


    model, tr = make_model_trainer(s, neptune_logger, par['learning_rate'])
    tr.fit(model, train_loader, test_loader)
    
    tr.save_checkpoint('models/{}/{}/trained_model.ckpt'.format(PARAMS['experiment_name'], datasetName))
    neptune_logger.experiment.log_artifact('models/{}/{}/trained_model.ckpt'.format(PARAMS['experiment_name'], datasetName))

run_experiment(dfh, 'dfh', PARAMS)
run_experiment(dfh_r, 'dfhr', PARAMS)
run_experiment(dfh_phi, 'dfhphi', PARAMS)
run_experiment(dfp, 'dfp', PARAMS)
run_experiment(dfp_r, 'dfpr', PARAMS)
run_experiment(dfp_phi, 'dfpphi', PARAMS)

