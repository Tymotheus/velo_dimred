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


#model = VeloAutoencoderLt(VeloEncoder(400), VeloDecoder(400), learning_rate=0.69)

model = VeloAutoencoderLt.load_from_checkpoint(os.path.join('models', 'Medium-net ReLU Adam', 'dfh', 'trained_model.ckpt' ))

print(model.lr)

reducedData = model.enc.forward(torch.tensor(dfh.values, dtype=torch.float))
reducedData = reducedData.detach().numpy()
print(reducedData)
