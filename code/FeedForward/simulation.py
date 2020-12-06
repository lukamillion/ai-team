import numpy as np
import pandas as pd
import torch
import multiprocessing
import progressbar
from progressbar import FormatLabel, Percentage, Bar, ETA

from data_loader import DataLoader



# TODO accept sim parameter dicts

class Agent():
    """
        An agent that has a netwerk stored. It walks based on the given neighbors one Time step.
    """
    def __init__(self, model, pos_vel_0, id=100, frame_0=0, FPS=16, truth_with_vel=False, device="cpu"):
        
        
        self.id = id
        
        self.model = model
        self.device = device
        self.truth_with_vel = truth_with_vel
        
        if self.device.type.startswith("cuda"):
            self.model.cuda()
        
        self.pos_vel_0 = pos_vel_0.copy()
        self.frame_c = frame_0
        
        self.FPS = FPS
        
        # position and velocity data
        self.frames = [frame_0]
        self.traj = [pos_vel_0]

                
    @property 
    def pos(self):
        # return the current postition of the agent
        return self.traj[-1][ :2]
    
    @property
    def vel(self):
        # return the current velocity of the agent
        return self.traj[-1][ 2:]
    
    
    def step(self, neighbors):
        """
            Perform one time step 

            PARAM:
                neighbors: flatend vector containing the n nn neighbors

            RETURN:
                traj: new position
        """
        with torch.no_grad(): # we do not want to train the model anymore
            x_sim = torch.from_numpy(neighbors).to(self.device) # create the input tensor on the device

            y_sim = self.model(x_sim.float())   # do the forward pass
        
        # compute the velocity if it is not predicted
        if self.truth_with_vel:
            v_sim = y_sim[2:]
            y_sim = y_sim[:2]
        else:
            v_sim = (y_sim[:2]-x_sim[:2])*self.FPS
        
        # we append the newo position to the trajectory
        self.traj.append( np.concatenate( (y_sim.cpu().detach().numpy(), v_sim.cpu().detach().numpy()) ) )
        self.frame_c +=1
        self.frames.append(self.frame_c)
        
        return self.traj[-1]
        




class Engine():
    def __init__(self, ds, agents=[], nn=10, stop_agent=False, xlim=350, mode="wraps", ret_vel=True, nn_vel=True,
                 truth_with_vel=True, downsample=1, exportpath="sim.csv"):
        
        self.ds = DataLoader(exportpath)
        self.ds.copy(ds)  # we copy the data so that we do not manipulate the original data
        
        self.agents = agents
        
        self.nn = nn
        self.ret_vel = ret_vel
        self.nn_vel = nn_vel
        self.truth_with_vel = truth_with_vel
        self.mode = mode
        self.downsample = downsample
        self.stop_agent = stop_agent
        self.xlim = xlim
        self.cur_f = 0
        
    def step(self, ):
        """
            moves every Agent one step

            PARAM:
                -
            RETURN:
                -
        """
        # get all person and agents in the frame
        in_frame, pos_vel_f = self.ds.frame(self.cur_f, ret_vel=self.ret_vel)

        # move every agent
        for a in self.agents:
            # we remove the agent if it exceeded a certain x position
            if a.pos[0]>self.xlim and self.stop_agent:
                continue
            
            # only concider agents that are in the current frame
            if a.id not in in_frame :
                continue
            
            # get the nn for the agent
            _, pos_velnn, _ = self.ds.get_nn(in_frame, pos_vel_f, np.where(in_frame==a.id)[0],
                                           self.nn,
                                           mode=self.mode   )
            
            # remove velocity if they are not used
            pos_vel = pos_velnn.copy()
            if ((not self.nn_vel) & self.ret_vel):
                pos_vel = np.concatenate((pos_vel[0,:], pos_vel[1:,0:2].ravel()))
            else:
                pos_vel = pos_vel.ravel()

#TODO implement frame override in dataloader
            # predict the next step
            n_pos_vel = a.step(pos_vel).copy()
            # we need to switch the x and y coordinates because we do a lowlevel acess
            n_pos_vel [[0, 1]] = n_pos_vel [[1, 0]]

            # build row to incert/ overwrite
            entry = np.concatenate( ([a.id], [self.cur_f+self.downsample], n_pos_vel[:2], [0], n_pos_vel[2:] ) )
            
            # check if we insert or override
            if len(self.ds.data[(self.ds.data["p"]==a.id) & (self.ds.data["f"]==self.cur_f+self.downsample)]):
                if self.downsample>1: print("WARNIG: you may have additional stepps in dset, -> please remove old agent befor simulating.")
                self.ds.data[(self.ds.data["p"]==a.id) & (self.ds.data["f"]==self.cur_f+self.downsample)] = [entry]
            else:
                self.ds.data = self.ds.data.append(pd.DataFrame([entry],
                                                                columns=list(self.ds.data)),
                                                               ignore_index=True)    
                
            
        # set the frame counter up
        self.cur_f += 1
        
    
    def run(self, start_f, stop_f, ):
        """
            Runs a simulation from given start frame to the stop frame.

            PARAM:
                start_f:    frame id where to start the simulaton
                stop_f:     frame id where to stop the simulation
        
            RETURN:
                -

        """
        self.cur_f = start_f
        print("sim from : {} to {}".format(start_f, stop_f))

        # init progressbar but flush first the stdout
        print('', end='', flush=True)
        widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar(), ' ', ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=stop_f)
        pbar.start()
        
        # place all agent in the dataset
        for  a in self.agents:

            pos = a.pos_vel_0[:2]
            entry = np.concatenate( ([a.id],  [a.frame_c], np.flip(pos) , [0],  a.pos_vel_0[2:]) )
            
            # we need to check if we have to override data or insert a new row
            if len(self.ds.data[(self.ds.data["p"]==a.id) & (self.ds.data["f"]==a.frame_c)]):
                self.ds.data[(self.ds.data["p"]==a.id) & (self.ds.data["f"]==a.frame_c)] = [entry]
            else:
                self.ds.data = self.ds.data.append(pd.DataFrame([entry],
                                                                columns=list(self.ds.data)),
                                                               ignore_index=True)
        
        
        while self.cur_f < stop_f:
            #do one step, eg move all agents
            self.step()
            #update progressbar
            widgets[0] = FormatLabel('frame: {:4}'.format(self.cur_f))
            pbar.update(self.cur_f)
        
        pbar.finish()


    
    def save(self, ):
        pass