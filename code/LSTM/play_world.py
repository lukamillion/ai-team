import torch

from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import numpy.linalg as la
import plotly.express as px


import data_loader as dL
import dataViewer as dV



class LSTM(nn.Module):
    """
    Use pytorch to define class for LSTM, has to be the same as the class defined in trainer.py

    """
    
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,batch_first=True)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
 
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
 
    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [batch_size, seq_len, num_directions*hidden_size]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view( self.batch_size, len(input[0]), -1), self.hidden)
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out)
        return y_pred.view(-1)










def find_nn(pos, env, nn):
    """
    pos ... positions of which to find nearest neighbours of (the position of the agent)
    env ... positions of all the other people
    nn ... number of nearest neighbours

    returns position of nearest neighbours

    """
    dist = env-pos
    dist = np.array([la.norm(dist[i]) for i in range(0, len(dist))])
    nn_index = dist.argsort()[:nn]
    return env[nn_index]


def create_input(loader, agent_pos, frame, nn):
    """
    loader ... loader object
    agent_pos ... position of agent
    frame ... which frame to look at
    nn ... number of nearest neighbours

    returns a vector to be used as input to the network

    """
    ids, pos = loader.frame(frame, ret_vel=False)
    x = find_nn(agent_pos, pos, nn)
    return x


ID = "180-140"
loader = dL.DataLoader("Datasets/UG/UG-roh_nachkorrigiert/ug-" + ID + ".txt")
loader.load()

model_ID = "180-140-5"

nene = 5

model_filename = "LSTM__ID-" + model_ID + ".model" 

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU
torch.no_grad()

model = torch.load(model_filename)

model.hidden = model.init_hidden()
batch_size = 1
model.batch_size= batch_size
seq_len = 1

model.hidden = model.init_hidden()




agent_id = 60 #choose which agent to replace
start_frame = 50 #choose which frame to start at
num_frames = 100 #how many frames to simulate


#generate agent traj
agent_traj = np.zeros((num_frames, 2))
temp_frames, temp_traj  = loader.person(agent_id, ret_vel=False)
loader.remove_person(agent_id)

agent_traj[0] = temp_traj[start_frame]

for t in range(0, num_frames-1):

    if t%10==0:
        print(t)
    
    x = create_input(loader, agent_traj[t], temp_frames[start_frame]+t, nene)
    x = x-agent_traj[t] #use relative coordinates
    x = torch.Tensor(x).view(batch_size,seq_len,-1)
    vel =  model(x).detach().numpy()
    flipped_vel = np.zeros_like(vel)
    flipped_vel[0] = vel[1]
    flipped_vel[1] = vel[0]
    agent_traj[t+1] = agent_traj[t] + flipped_vel/16

    
loader.append_person(-1, np.arange(temp_frames[start_frame],temp_frames[start_frame]+num_frames), agent_traj)

#plotting
dV.plotTraj(loader,  boundaries=[-600, 600, -400, 400],
         people=None,
         ai=[-1],
         legend=False,
         title="Trajectories",
         path="trajectories.png",
         save=False)

  
