#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 19:48:31 2020

@author: alexanderjurgens
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  

data=np.genfromtxt( r'/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/AO/ao-240-400_combine.txt', names=True,skip_footer=1,dtype=None,delimiter=' ')

a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/positions.pkl", "rb")
positions = pickle.load(a_file)
a_file.close()
a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/people.pkl", "rb")
people = pickle.load(a_file)
a_file.close()
a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/velocities.pkl", "rb")
velocities = pickle.load(a_file)
a_file.close()

frames=np.arange(max([people[k][-1][1] for k in people]))



neighs=6



data=[]
for frame in positions:
    if frame>40 and np.linalg.norm(positions[frame]!=10000)**2>2*neighs:
        for person in people:
            if positions[frame][person][0]!=10000 and positions[frame-1][person][0]!=10000 and positions[frame+1][person][0]!=10000:
                order=np.argsort(np.linalg.norm(np.subtract(positions[frame].copy(),positions[frame][person].copy()),axis=1))
                others=positions[frame][order] #list of all positions ordered by distance to person
                otherv=velocities[frame][order]
                npos=others[1:neighs+1].flatten()
                nvel=otherv[1:neighs+1].flatten()
                information = np.concatenate((np.array([positions[frame][person],velocities[frame][person]]).flatten(),npos,nvel))
                result      = positions[frame+1][person]
                data.append((information,result))

a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/bottleneck_training_data_"+str(neighs)+".pkl", "wb")
positions = pickle.dump(data,a_file)
a_file.close()
print("\n\n----------------- DONE -----------------")
        