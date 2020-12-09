#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:52:25 2020

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


def dist(i,j,frame):
    if i%10==0and j==0:
        print (frame,i)
    try:
        index1=np.where(np.array([dat[1] for dat in people[i]])==frame)[0][0]
        index2=np.where(np.array([dat[1] for dat in people[j]])==frame)[0][0]
        return np.linalg.norm([people[i][index1][2]-people[j][index2][2],people[i][index1][3]-people[j][index2][3]])
    except:
        return None

for p in range(1,200,1):
    plt.plot([dat[2] for dat in people[p]],[dat[3]  for dat in people[p]],linewidth=0.5)
plt.savefig('/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/bottleneck_trajectories.png',dpi=800)
plt.show()

frame=500

xs=np.array([pos[0] for pos in positions[frame]])
ys=np.array([pos[1] for pos in positions[frame]])
v1=np.array([vel[0] for vel in velocities[frame]])
v2=np.array([vel[1] for vel in velocities[frame]])

v1=v1[xs!=10000]
v2=v2[xs!=10000]
xs=xs[xs!=10000]
ys=ys[ys!=10000]

plt.quiver(xs,ys,v1,v2)
plt.show()

plt.scatter(xs,ys)
plt.show()

def animateandsafeasgif(positions):
    fig, ax = plt.subplots()  
    x, y = [], []
    ln1, = plt.plot([], [], 'ro')  
    def init():  
        ax.set_xlim(-250,450)  
        ax.set_ylim(-650, 850)  
      
    def update(frame):  
        if frame%100==0:
            print(frame)
        x=[pos[0] for pos in positions[frame]]
        y=[pos[1] for pos in positions[frame]]
        ln1.set_data(x, y)  
    ani = FuncAnimation(fig, update, np.arange(0,max([people[k][-1][1] for k in people])), init_func=init)  
    plt.show()
    writer = PillowWriter(fps=25)  
    ani.save("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/data_animation.gif", writer=writer)  
def quiverandsafeasgif(positions,velocities):
    x,y,u,v=[],[],[],[]    
    fig, ax = plt.subplots(1,1)
    
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    def init():  
        ax.set_xlim(-250,450)  
        ax.set_ylim(-650, 850)  

    def update(frame):
        if frame%100==0:
            print(frame)
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """
        ax.clear()
        ax.set_xlim(-250,450)  
        ax.set_ylim(-650, 850)  
        x=np.array([pos[0] for pos in positions[frame]])
        y=np.array([pos[1] for pos in positions[frame]])
        u=np.array([vol[0] for vol in velocities[frame]])
        v=np.array([vol[1] for vol in velocities[frame]])
    
        ax.quiver(x,y,u,v)
    
    ani = FuncAnimation(fig, update, frames , init_func=init)  
    writer = PillowWriter(fps=25)  
    ani.save("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/quiver_animation.gif", writer=writer)  

quiverandsafeasgif(positions, velocities)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# neighs=6 #number of neighbors to be considered by the agent
# input_size = 4+neighs*4  #two 2d vectors for own position and velocity, same for each considered neighbor
# hidden_layers = [40,40,10] 
# output_size = 2

# model = nn.Sequential(
#     nn.Linear(input_size, hidden_layers[0]),
#     nn.ReLU(),
#     nn.Linear(hidden_layers[0], hidden_layers[1]),
#     nn.ReLU(),
#     nn.Linear(hidden_layers[1], hidden_layers[1]),
#     nn.ReLU(),
#     nn.Linear(hidden_layers[2], output_size),
#     nn.LogSoftmax(dim=1)
# )
# criterion = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.003)
# epochs = 5
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in trainloader:
#         # Flatten the Image from 28*28 to 784 column vector
#         images = images.view(images.shape[0], -1)
        
#         # setting gradient to zeros
#         optimizer.zero_grad()        
#         output = model(images)
#         loss = criterion(output, labels)
        
#         # backward propagation
#         loss.backward()
        
#         # update the gradient to new gradients
#         optimizer.step()
#         running_loss += loss.item()
#     else:
#         print("Training loss: ",(running_loss/len(trainloader)))
        
        
  