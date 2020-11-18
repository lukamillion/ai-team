#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:40:32 2020

@author: alexanderjurgens
"""

from Neuralnet import *
import torch

a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/bottleneck_training_data.pkl", "rb")
data = pickle.load(a_file)
a_file.close()

myboi=Net()
try:
    myboi.teach(data,epochs=1000,strech=3000)
    torch.save(myboi,'/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/the_latest_network.pt')
except:
    torch.save(myboi,'/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/the_latest_network.pt')














































































