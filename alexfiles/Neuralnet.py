#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:24:11 2020

@author: alexanderjurgens
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset,DataLoader

neighs=6
batch_size=1475

a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/bottleneck_training_data.pkl", "rb")
data = pickle.load(a_file)
a_file.close()

class Net(nn.Module):

    def __init__(self, neighs=6):
        inputt=(neighs+1)*4
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputt, 28)
        self.fc2 = nn.Linear(28, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 2)
        self.neighs=neighs
        self.strech=1

    def forward(self, x,isarray=False):
        if isarray:
            x=torch.from_numpy(x).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return torch.sigmoid(self.fc5(x))

    def teach(self,data,epochs=1000,silent=False,showperformance=True,strech=3000):
        self.strech=strech
        train=Trainingset(data,strech=strech)
        loader=DataLoader(train,batch_size=batch_size)
        optimizer=optim.Adam(self.parameters(),lr=0.002)
        loss_fn = nn.MSELoss()
        if showperformance:
                self.eval()
                error=0
                for batch in loader:
                    inputs,targets=batch
                    error+=torch.norm(targets-self(inputs))**2
                print("Null-performance of "+str(error**0.5))

        for epoch in range(epochs):
            if not silent and epoch%10==0: #and epoch%100!=0:
                print("Epoch "+str(epoch)+" out of "+str(epochs)+" ")
            self.train()
            for batch in loader:
                optimizer.zero_grad()
                inputs,targets=batch
                output=self(inputs)
                loss=loss_fn(output,targets)
                loss.backward()
                optimizer.step()
            if showperformance:
                self.eval()
                error=0
                for batch in loader:
                    inputs,targets=batch
                    error+=torch.norm(targets-self(inputs))**2
                print("Performance of "+str(error.detach()**0.5))
        if not silent:
            print("Training Done!")
        return self

class Trainingset(Dataset):
    def __init__(self,data,strech=2000):
        self.data=data
        self.strech=strech
    def __len__(self):
        return len(self.data)
    def __getitem__(self,k):
        strech=self.strech
        sample=[torch.from_numpy(self.data[k][0]/strech+0.5).float(),torch.from_numpy(self.data[k][1]/strech+0.5).float()]
        return sample
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
