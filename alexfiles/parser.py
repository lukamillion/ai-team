#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:45:17 2020

@author: alexanderjurgens
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  

data=np.genfromtxt( r'/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/AO/ao-240-400_combine.txt', names=True,skip_footer=1,dtype=None,delimiter=' ')

people={}     #dictionary of people. people[k] is the data about the k-th person in the format np.array[(k,frame,x,y,z),...]
positions={}  #dictionary of positions. positions[frame] is an array of all positions at frame 'frame' in the format 
i0=0
for i,dat in enumerate(data):
    if i!=0 and dat[0]!=data[i-1][0]:
        people[data[i-1][0]-1]=data[i0:i]
        i0=i
frames=np.arange(max([people[k][-1][1] for k in people]))

alternative=np.array([10000,10000])
for frame in frames:
    pos=[]
    for k in people:
        if frame in [dat[1] for dat in people[k]]:
            index=np.where(np.array([dat[1] for dat in people[k]])==frame)[0][0]
            pos.append(np.array([people[k][index][2],people[k][index][3]]))
        else:
            pos.append(alternative)
    positions[frame]=np.array(pos)
        
velocities={}
for frame in positions:
    velocities[frame]=positions[frame].copy()
for i,vec in enumerate(velocities[0]):
    velocities[0][i]=np.array([0,0])

for frame in positions:
    if frame!=0:
        for k,pos in enumerate(positions[frame]):
            if np.array_equal(pos,alternative) or np.array_equal(positions[frame-1][k],alternative):
                velocities[frame][k]=np.array([0,0])
            else:
                velocities[frame][k]=pos-positions[frame-1][k]
   

a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/velocities.pkl", "wb")
pickle.dump(velocities, a_file)
a_file.close()

a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/people.pkl", "wb")
pickle.dump(people, a_file)
a_file.close()

a_file = open("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/positions.pkl", "wb")
pickle.dump(positions, a_file)
a_file.close()
print('hi')
        
        
def dist(i,j,frame):
    if i%10==0and j==0:
        print (frame,i)
    try:
        index1=np.where(np.array([dat[1] for dat in people[i]])==frame)[0][0]
        index2=np.where(np.array([dat[1] for dat in people[j]])==frame)[0][0]
        return np.linalg.norm([people[i][index1][2]-people[j][index2][2],people[i][index1][3]-people[j][index2][3]])
    except:
        return None

