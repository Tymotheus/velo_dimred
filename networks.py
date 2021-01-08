#!/usr/bin/env jupyter

#
# Splits:
# Dimentionality: [t, module, channel]
#
# Entire detector: x=t [module * channel, 1] - to little data
# Only channels: x=t*module [channel, 1]
# Only phi/R x=t*module/2 [channel, 1]

# https://gitlab.cern.ch/mmajewsk/calina.git
import sys

sys.path.append("../calina/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from calibration_dataset import Tell1Dataset


class MyDS(Tell1Dataset):
    filename_format = "%Y-%m-%d"
    filename_regex_format = r"\d{4}-\d{2}-\d{2}.csv"


def get_dataset(path):
    data_list = MyDS.get_filepaths_from_dir(datapath)
    mds = MyDS(data_list, read=True)
    return mds


class VeloEncoder(nn.Module):
    def __init__(self, input_size):
        nn.Module.__init__(self)
        self.e1 = nn.Linear(input_size, 40)
        self.e2 = nn.Linear(40, 10)
        self.e3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.e1(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.e2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.e3(x)
        return x


class VeloDecoder(nn.Module):
    def __init__(self, output_size):
        nn.Module.__init__(self)
        self.e3 = nn.Linear(2, 10)
        self.e2 = nn.Linear(10, 40)
        self.e1 = nn.Linear(40, output_size)

    def forward(self, x):
        x = self.e3(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.e2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = self.e1(x)
        return x

class VeloAutoencoderLt(pl.LightningModule):
    def __init__(self, encoder, decoder, learning_rate):
        pl.LightningModule.__init__(self)
        self.enc = encoder
        self.dec = decoder
        self.lr = learning_rate

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #lambda1 = lambda epoch: 0.95 ** epoch
        #scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=lambda1 )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.2,
                                                               patience=2,
                                                               min_lr=1e-6,
                                                               verbose=True)
        return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'monitor' : 'loss'
    }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss
