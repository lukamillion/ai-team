#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 23:05:31 2020

@author: alexanderjurgens
"""

import torch
import pickle
from Neuralnet import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  

inputt=28

class agent:
    def __init__(self,net,pos0,number):
        self.no=number
        self.net=net
        self.pos=pos0
        self.neighs=net.neighs
        self.last=pos0
    def step(self,posses,vels):
        if self.pos[1]<-500 or self.pos[1]==10000:
            self.pos=np.array([10000,10000])
            posses[self.no]=self.pos
            vels[self.no]=0
            return self.pos
        order=np.argsort(np.linalg.norm(np.subtract(posses.copy(),self.pos),axis=1))
        others=posses[order]
        otherv=vels[order]
        npos=others[1:self.neighs+1].flatten()
        nvel=otherv[1:self.neighs+1].flatten()
        information = np.concatenate((np.array([self.pos,self.pos-self.last]).flatten(),npos,nvel))/self.net.strech+0.5
        self.last=self.pos
        self.pos=np.array(self.net.forward(information, isarray=True).detach())*self.net.strech-self.net.strech/2.0
        # posses[self.no]=self.pos
        # vels[self.no]=self.pos-self.last
        return self.pos
        

# net = Net()
# net.load_state_dict(torch.load("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/trained_network.pt"))
net=torch.load("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/the_latest_network.pt")
net.eval()
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

def minframe(person):
    return np.argmax(np.array([positions[frame][person][0] for frame in positions])!=10000)+1

minframes=[minframe(person) for person in people]

agents={}
for fuckingpeasant in people:
    if fuckingpeasant%3==0:
        agents[fuckingpeasant]=agent(net,positions[minframes[fuckingpeasant]][fuckingpeasant],fuckingpeasant)

def animateandsafeasgif(positions,velocities):
    fig, ax = plt.subplots()  
    ln1, = plt.plot([], [], 'ro')  
    ln2, = plt.plot([], [], 'ko')  
    def init():  
        ax.set_xlim(-250,450)  
        ax.set_ylim(-650, 850)  
      
    def update(frame):  
        if frame%100==0:
            print(frame)
        poslast,vellast=positions[frame-1],velocities[frame-1]
        posnow,velnow=positions[frame],velocities[frame]
        for k in people:
            if k in agents and minframes[k]<=frame:
                pos=agents[k].step(poslast.copy(),vellast.copy())
                posnow[k]=pos
                velnow[k]=pos-poslast[k]
                # velocities[frame+1][k]=pos-positions[frame-1][k]
                # positions[frame+1][k]=pos
            else:
                None
        x=np.array([pos[0] for pos in positions[frame]])
        y=np.array([pos[1] for pos in positions[frame]])
        ags=np.array([(k in agents) for k,pos in enumerate(x)])
        ln1.set_data(x[np.invert(ags)], y[np.invert(ags)])  
        ln2.set_data(x[ags],y[ags])
    ani = FuncAnimation(fig, update, np.arange(30,max([people[k][-1][1] for k in people])), init_func=init)  
    plt.show()
    writer = PillowWriter(fps=25)  
    ani.save("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/tenpeople.gif", writer=writer)  

animateandsafeasgif(positions,velocities)

zeroframe=50
posses=positions[zeroframe]
vels=velocities[zeroframe]
agents={}
for k in people:
    if positions[zeroframe][k][0]!=10000 and positions[zeroframe-1][k][0]!=10000:
        agents[k]=agent(net,posses[k],k)

# fig, ax = plt.subplots() 
# ax.set_xlim(-250,450)  
# ax.set_ylim(-650, 850)  
# plt.scatter(posses[:,0],posses[:,1])
# plt.show()

# for k in agents:
#     agents[k].walk(posses, vels)
    
# fig, ax = plt.subplots() 
# ax.set_xlim(-250,450)  
# ax.set_ylim(-650, 850)  
# plt.scatter(posses[:,0],posses[:,1])
# plt.show()
# for k in agents:
#     agents[k].walk(posses, vels)
    
# fig, ax = plt.subplots() 
# ax.set_xlim(-250,450)  
# ax.set_ylim(-650, 850)  
# plt.scatter(posses[:,0],posses[:,1])
# plt.show()
# for k in agents:
#     agents[k].walk(posses, vels)
    
# fig, ax = plt.subplots() 
# ax.set_xlim(-250,450)  
# ax.set_ylim(-650, 850)  
# plt.scatter(posses[:,0],posses[:,1])
# plt.show()
# for k in agents:
#     agents[k].walk(posses, vels)
    
# fig, ax = plt.subplots() 
# ax.set_xlim(-250,450)  
# ax.set_ylim(-650, 850)  
# plt.scatter(posses[:,0],posses[:,1])
# plt.show()
# for k in agents:
#     agents[k].walk(posses, vels)
    
# fig, ax = plt.subplots() 
# ax.set_xlim(-250,450)  
# ax.set_ylim(-650, 850)  
# plt.scatter(posses[:,0],posses[:,1])
# plt.show()
# for k in agents:
#     agents[k].walk(posses, vels)
    
# fig, ax = plt.subplots() 
# ax.set_xlim(-250,450)  
# ax.set_ylim(-650, 850)  
# plt.scatter(posses[:,0],posses[:,1])
# plt.show()


# def animateandsafeasgif(posses,vels):
#     fig, ax = plt.subplots()  
#     #ln1, = plt.plot([], [], 'ro')  
#     ln2, = plt.plot([], [], 'ko')  
#     def init():  
#         ax.set_xlim(-250,450)  
#         ax.set_ylim(-650, 850)  
      
#     def update(frame):  
#         if frame%100==0:
#             print(frame)
#         for k in agents:
#             p0=agents[k].step(posses,vels)
#             vels[k]=p0-posses[k]
#             posses[k]=p0
#         x=posses[:,0]
#         y=posses[:,1]
#         plt.title(str(frame*1.0/25)+"s")
#         #ags=np.array([(k in agents) for k,pos in enumerate(x)])
#         #ln1.set_data(x[np.invert(ags)], y[np.invert(ags)])  
#         ln2.set_data(x,y)
#     ani = FuncAnimation(fig, update, np.arange(zeroframe,100), init_func=init)  
#     plt.show()
#     writer = PillowWriter(fps=25)  
#     ani.save("/Users/alexanderjurgens/Desktop/MyIDE/Bottleneck/collective.gif", writer=writer)  

        




