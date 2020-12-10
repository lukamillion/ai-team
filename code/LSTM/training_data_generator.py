import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import data_loader as dL
import dataViewer as dV



def find_in_array(data, arr):
    """
    data ... data to be found
    arr ... array in which to search for data
    returns position of the data in the array or -1 if data is not found
    """
    for i in range(0, len(arr)):
        if np.array_equal(data,arr[i]):
            return i
        else:
            return -1
        
    


def generate_training_data(loader, xy_limit, nn, seq_len, downsample_num):
    """
    loader ... loader object
    xy_limit ... region of interest
    nn ... number of nearest neighbours
    seq_len ... length for one sequence (all sequences of other lengths will be discarded)
    downsample_num ... truth values predicts the n-th frame with n being downsample_num, with 1 being no downsampling
    
    returns x the inputs and y the truth values for the training set
    x and y are array of sequences of shape (m, seq_len, 2*nn) and (m, seq_len, 2)
    """

    
    x = []
    y = []

    plot = 0
    place = 0
    num_people = loader.persons

    for p in range(1, num_people+1):
        pers_frames, traj = loader.person(p, ret_vel=False)
        traj_len = len(traj)
        environments = []
        solutions = []
        
        #plt.plot(traj[:,0], traj[:,1], linestyle = "none", marker = ".")
        #if p>=50:    
        #    plt.show()
        #    sys.exit()
        frame = 0
        while (xy_limit[0]>=traj[frame,0] or traj[frame,0]>=xy_limit[1] or xy_limit[2]>=traj[frame,1] or traj[frame,1]>=xy_limit[3]) and frame<traj_len-1-downsample_num:
            frame+=1
            
        while xy_limit[0]<=traj[frame,0]<=xy_limit[1] and xy_limit[2]<=traj[frame,1]<=xy_limit[3] and frame<traj_len-prediction_length-downsample_num:

            ids, pos = loader.frame(pers_frames[frame], ret_vel=False)
            nn_ids, nn_pos = loader.frame_nn(pers_frames[frame],p,nn,use_roi=False, ret_vel=False)
            nn_pos = nn_pos[:,:2]
            nn_pos = nn_pos-nn_pos[0]
            #append all relative positions of neighbours
            to_append = nn_pos[1:,:].reshape((nn*2))
            environments.append(to_append)

            to_append = traj[frame+downsample_num]
            solutions.append(to_append)
            
            frame += 1
            
        if len(environments)>=seq_len:
            x.append(environments[:seq_len])
            y.append(solutions[:seq_len])
        
    return x, y
                


def format_for_saving(nn,seq_len,x,y):
    """
    nn ... number of nearest neighbours in dataset
    seq_len ... sequence length
    x,y ... outputs of generate_training_data

    """
    
    sit_len = nn*2+2
    second_dim = (nn*2+2)*seq_len
    first_dim = len(y)
    data = np.zeros((first_dim,second_dim))

    for i in range(first_dim):
        to_add = np.zeros(second_dim)
        for j in range(seq_len):
            to_add[j*sit_len:(j+1)*sit_len-2] = x[i][j] 
            to_add[(j+1)*sit_len-2:(j+1)*sit_len] = y[i][j]
        data[i] = to_add
    return data



ID = "180-140"
loader = dL.DataLoader("Datasets/UG/UG-roh_nachkorrigiert/ug-" + ID + ".txt")
loader.load()

xy_limit = [-200,200,50,110]
#xy_limit = [-600,600,-100,300]
prediction_length = 1
nn = 5
seq_len = 50

x, y = generate_training_data(loader, xy_limit, nn, seq_len, 16)

data = format_for_saving(nn,seq_len,x,y)
print("Shape of data: ", data.shape)


filename = "./training_data/training_set-" + ID + "-%d" %nn
np.savetxt(filename, data, delimiter = ",")
print("Data Saved in file: ", filename)
