#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
import pytorch_lightning as pl
import torch
import logging
logging.basicConfig(level=logging.INFO)
import os

from calibration_dataset import Tell1Dataset

PARAMS = {'max_epochs': 1,
          'learning_rate': 0.05,
          'batch_size': 64,
          'gpus' : 1,
          'experiment_name' : 'debuging standarized SGD no-dropout huge-batches relu',
          'tags' : ["testing"],
          'source_files' : ['analyze_Pawel.py', 'networks.py']
         }

datasetNames = ['dfh', 'dfhr', 'dfhphi', 'dfp', 'dfpr', 'dfpphi']


class MyDS(Tell1Dataset):
	filename_format = '%Y-%m-%d'
	filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'

#loading the data
datapath = os.path.join("data", "calibrations")
data_list = MyDS.get_filepaths_from_dir(datapath)
mds = MyDS(data_list, read=True)


dfh = mds.dfh.df.iloc[:,9:]
dfh_r = mds.dfh['R'].df.iloc[:,9:]
dfh_phi = mds.dfh['phi'].df.iloc[:,9:]
dfp = mds.dfp.df.iloc[:,9:]
dfp_r = mds.dfp['R'].df.iloc[:,9:]
dfp_phi = mds.dfp['phi'].df.iloc[:,9:]

dfh_metadata = mds.dfh.df.iloc[:,:9]
dfh_r_metadata = mds.dfh['R'].df.iloc[:,:9]
dfh_phi_metadata = mds.dfh['phi'].df.iloc[:,:9]
dfp_metadata = mds.dfp.df.iloc[:,:9]
dfp_r_metadata = mds.dfp['R'].df.iloc[:,:9]
dfp_phi_metadata = mds.dfp['phi'].df.iloc[:,:9]

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def plot(dataset, datasetName, metadata, exp_name, exp_id):
    model_path = os.path.join('models', exp_name, datasetName, 'trained_model.ckpt')

    if not os.path.exists(model_path):
        logging.info('{} does not exists, exiting'.format(model_path) )
        exit()

    model = torch.load(model_path)

    reducedData = model.enc.forward(torch.tensor(dfh.values, dtype=torch.float))
    reducedData = reducedData.detach().numpy()

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
    cmap = get_cmap( len(sensorNumberList) )
    for sensorNumber in range( len(sensorNumberList) ):
        plt.scatter(x2DList[sensorNumber],y2DList[sensorNumber], c =cmap(sensorNumber) , edgecolor='none', alpha=alpha, label=sensorNumberList[sensorNumber] )

    plt.xlabel('Reduced variable 1')
    plt.ylabel('Reduced variable 2')
    plt.legend(title="Module nr.", ncol=5)
    plt.show()
    fig.savefig( os.path.join('models', exp_name, datasetName,'reduced.png') )
    
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
