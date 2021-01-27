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
dfh_sensor_numbers = mds.dfh.df.iloc[:,1]
#print('mds.dfh.df')
#print(mds.dfh.df)
#print('dfh_sensor_numbers')
#print(dfh_sensor_numbers)

#print('for sensor_key in dfh:')
#for sensor_key in dfh:
#    print(dfh[sensor_key])
dfh_r = mds.dfh['R'].df.iloc[:,9:]
#print(mds.dfh['R'].df)
dfh_phi = mds.dfh['phi'].df.iloc[:,9:]
#print(mds.dfh['phi'].df)
dfp = mds.dfp.df.iloc[:,9:]
#print('mds.dfp.df')
#print(mds.dfp.df)
dfp_r = mds.dfp['R'].df.iloc[:,9:]
#print(mds.dfp['R'].df)
dfp_phi = mds.dfp['phi'].df.iloc[:,9:]
#print(mds.dfh['phi'].df)

dfh_metadata = mds.dfh.df.iloc[:,:9]
dfh_r_metadata = mds.dfh['R'].df.iloc[:,:9]
dfh_phi_metadata = mds.dfh['phi'].df.iloc[:,:9]
dfp_metadata = mds.dfp.df.iloc[:,:9]
dfp_r_metadata = mds.dfp['R'].df.iloc[:,:9]
dfp_phi_metadata = mds.dfp['phi'].df.iloc[:,:9]

'''
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

PARAMS = {'max_epochs': 1,
          'learning_rate': 0.02,
          'batch_size': 32,
          'gpus' : 1,
          'experiment_name' : 'testing',
          'tags' : ["testing"],
          'source_files' : ['analyze_Pawel.py', 'networks.py']  
         }



datasetNames = ['dfh', 'dfhr', 'dfhphi', 'dfp', 'dfpr', 'dfpphi']

# for d in datasetNames:
#     if not os.path.exists('models\{}\{}'.format(PARAMS['experiment_name'], d)):
#         os.makedirs('models\{}\{}'.format(PARAMS['experiment_name'], d))

#trying to use os path join
for d in datasetNames:
    if not os.path.exists(os.path.join('models', PARAMS['experiment_name'], d)):
        os.makedirs(os.path.join('models', PARAMS['experiment_name'], d))


def make_model_trainer(s, neptune_logger, lr):
    s = 2048
    dec = VeloDecoder(s)
    enc = VeloEncoder(s)
    model = VeloAutoencoderLt(enc, dec, lr)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tr = pl.Trainer(logger=neptune_logger, callbacks=[lr_monitor],  max_epochs=PARAMS['max_epochs'], gpus=PARAMS['gpus'])
    return model, tr

plt.rcParams['figure.figsize'] = [16, 9*15]
procentage_or_num_of_comp = 2

'''
def scatter_data(single_data): 
    alpha = 0.4
    for sensor_data_key in single_data:
        dataset = single_data[sensor_data_key]
        dataset_after_reduction = full_pca(dataset,procentage_or_num_of_comp)
        scatter = plt.scatter(dataset_after_reduction.iloc[:,0], dataset_after_reduction.iloc[:,1], edgecolor='none', alpha=alpha,label=sensor_data_key)
        plt.legend(title="Module nr.")

def draw_a_plot(sensor, mod_key):
    for sensor_key in sensor:
        single_data = sensor[sensor_key]
        plot = plt.subplot(draw_a_plot.position, title=f'{mod_key} - {sensor_key}')
        set_color(plot)
        single_data = {k: v.drop('sensor',axis=1) for k, v in single_data.groupby('sensor')}
        scatter_data(single_data)
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        draw_a_plot.position+=1
'''


def run_experiment(dataset, datasetName, sensor_numbers, par):
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
    
    reduced_data = model.enc.forward(torch.tensor(dataset.values, dtype=torch.float))
    reduced_data = reduced_data.detach().numpy()

    current_sensor = 0
    for i_sensor in range( len(sensor_numbers) ):
        
    # tr.save_checkpoint('models\{}\{}\trained_model.ckpt'.format(PARAMS['experiment_name'], datasetName))
    # neptune_logger.experiment.log_artifact('models\{}\{}\trained_model.ckpt'.format(PARAMS['experiment_name'], datasetName))

	#implementing os path join
    tr.save_checkpoint(os.path.join('models', PARAMS['experiment_name'], datasetName,"trained_model.ckpt" ))
    neptune_logger.experiment.log_artifact(os.path.join('models', PARAMS['experiment_name'], datasetName,"trained_model.ckpt" ))
'''

import logging
import os
import torch.nn as nn
from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt

logging.basicConfig(level=logging.INFO)
PARAMS = {'max_epochs': 1,
          'learning_rate': 0.05,
          'batch_size': 32,
          'gpus' : 1,
          'experiment_name' : 'relu sigmoid-last small-net',
          'tags' : ["testing"],
          'source_files' : ['analyze_Pawel.py', 'networks.py']  
         }

datasetNames = ['dfh', 'dfhr', 'dfhphi', 'dfp', 'dfpr', 'dfpphi']

def plot(dataset, datasetName, metadata, exp_name, exp_id):
    model_path = os.path.join('models', exp_name, datasetName, 'trained_model.ckpt')
    print('model_path')
    print(model_path)
    if not os.path.exists(model_path):
        logging.info('{} does not exists, exiting'.format(model_path) )
        exit()

    model = torch.load(model_path)

    print(model.lr)

    reducedData = model.enc.forward(torch.tensor(dfh.values, dtype=torch.float))
    reducedData = reducedData.detach().numpy()
    print('dfh.values')
    print(dfh.values)    
    #print('reducedData')
    #for i in range(len(reducedData)):
    #    print( str(reducedData[i][0])+' '+str(reducedData[i][1]))
    
    x2DList = []
    y2DList = []
    sensorNumberList = [0]
   
    tempSensor = 0
    counter = 0
    tempX = []
    tempY = []    
    for sensor in metadata['sensor']:
        if int(sensor)==tempSensor:
            tempX.append(reducedData[counter][0])
            tempY.append(reducedData[counter][1])
        else:
            x2DList.append(tempX)
            y2DList.append(tempY)
            tempX = [reducedData[counter][0]]
            tempY = [reducedData[counter][1]]
            sensorNumberList.append( int(sensor) )
        counter = counter + 1
        tempSensor = sensor
    x2DList.append(tempX)
    y2DList.append(tempY)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    alpha = 0.4
    for sensorNumber in range( len(sensorNumberList) ):
        plt.scatter(x2DList[sensorNumber],y2DList[sensorNumber], edgecolor='none', alpha=alpha, label=sensorNumberList[sensorNumber] )
    #print('sensorNumberList')
    #print(sensorNumberList)
    #print('reducedData')
    #for i in range(len(reducedData)):
    #    print ( str(reducedData[i][0])+' '+str(reducedData[i][1]) )
    plt.xlabel('Reduced variable 1')
    plt.ylabel('Reduced variable 2')
    #plt.legend(title="Module nr.")
    plt.show()
    plt.savefig( os.path.join('models', exp_name, datasetName,'reduced.png') )
    
    #project = neptune.init('pawel-drabczyk/velodimred')
    #my_exp = project.get_experiments(id=exp_id)[0]

    #my_exp.append_tag('plot-added')
    #log_chart('matplotlib-interactive', fig, my_exp)

plot(dfh, 'dfh', dfh_metadata, PARAMS['experiment_name'], 'VEL-335')

#run_experiment(dfh, 'dfh', dfh_sensor_numbers, PARAMS)
#run_experiment(dfh_r, 'dfhr', PARAMS)
#run_experiment(dfh_phi, 'dfhphi', PARAMS)
#run_experiment(dfp, 'dfp', PARAMS)
#run_experiment(dfp_r, 'dfpr', PARAMS)
#run_experiment(dfp_phi, 'dfpphi', PARAMS)
