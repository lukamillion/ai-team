import numpy as np
import pandas as pd
import torch
import multiprocessing
import progressbar
from progressbar import FormatLabel, Percentage, Bar, ETA

from data_loader import DataLoader





class Agent():
    """
        An agent that has a netwerk stored. It walks based on the given neighbors one Time step.
    """
    def __init__(self, model, pos_vel_0, id=1000, frame_0=0, param=None, FPS=16, truth_with_vel=False, device="cpu"):
        """
            creats an Agent with an pre trained model. 

            PARAM:
                model:  pretrained torch model
                pos_vel_0: start position and velocity
                id:     agent id
                frame_0: start frame
                param:  if param is set all following parameters are ignored 
                FPS:    frame rate of dataset
                truth_with_vel: if true the velocity is predicted
                device: device to run sim on

            RETURN:
                -

        """
        
        self.id = id
        
        self.model = model
        if param is None:
            self.device = device
            self.truth_with_vel = truth_with_vel
            self.FPS = FPS

        else:
            self.device = param['device']
            self.truth_with_vel = param['dataset']['truth_with_vel']
            self.FPS = param['dataset']['fps']/param['dataset']['downsample']
            # we set the start frame to a integer multiple of downsample
            frame_0=int(frame_0/param['dataset']['downsample'])*param['dataset']['downsample']
        
        if self.device.type.startswith("cuda"):
            self.model.cuda()
        
        self.pos_vel_0 = pos_vel_0.copy()
        self.frame_c = frame_0
        
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
    def __init__(self, ds, agents=[], stop_agent=False, xlim=350, truth=None, param=None, ):
        """
            start a simulation for multiple agents

            PARAM:
                ds:         dataset with persons
                agents:     list of agents
                stop_agent: if True we stop the agents at xlim
                xlim:       limit for simulation in x direction
                truth:      original data may be used for further analysis
                param:      parameter dict of current experiment

            RETURN:
                -
        """        
        self.ds = DataLoader(None)
        self.ds.copy(ds)  # we copy the data so that we do not manipulate the original data

        self.t_ds = truth
        
        self.agents = agents
        
        self.nn = param['dataset']['neighbors']
        self.ret_vel = param['dataset']['ret_vel']
        self.nn_vel = param['dataset']['nn_vel']
        self.truth_with_vel = param['dataset']['truth_with_vel']
        self.mode = param['dataset']['mode']
        self.downsample = param['dataset']['downsample']

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

            # predict the next step
            n_pos_vel = a.step(pos_vel).copy()
            
            # insert step into dataobject
            self.ds.insert_row( a.id, self.cur_f+self.downsample, n_pos_vel, self.downsample)


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
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=stop_f+len(self.agents))
        pbar.start()
        
        # place all agent in the dataset
        for  a in self.agents:
            # insert step into dataobject
            self.ds.insert_row( a.id, a.frame_c, a.pos_vel_0, self.downsample)

        
        while self.cur_f < stop_f:
            #do one step, eg move all agents
            self.step()
            #update progressbar
            widgets[0] = FormatLabel('frame: {:4}'.format(self.cur_f))
            pbar.update(self.cur_f)
        
        # interpolate the 
        for j, a in enumerate(self.agents):
            widgets[0] = FormatLabel('interpolate: {:4}'.format(a.id))
            pbar.update(stop_f+j)
            self.ds.interpolate_person(a.id)

        widgets[0] = FormatLabel('Done!')
        pbar.finish()


    
    def save(self, name="sim.csv", include_truth=True):
        """
            Save the simulation in the dataloader compatilbe format.
            Make shure no agent id overlaps with the person id

            PARAM:
                name: name of export
                include_truth: if true the og dataset is included in the export
            RETURN: 
                -
        """
        
        # create a temporary dataloader object because other wise the simulation data
        # will have the truth values as well.
        ds_s = DataLoader(None)
        ds_s.copy(self.ds)

        # insert the original data to the save file
        if include_truth and self.t_ds is not None:
            to_copy = np.setdiff1d(self.t_ds.data['p'].unique(), ds_s.data['p'].unique())
            
            if len(to_copy)<len(self.t_ds.data['p'].unique()):
                print( "Warning! Only new IDs are considered.")# give a warning if not all data is copied
            
            for p in to_copy:   # loop over all people to copy
                frames_b, pos_vel_b =  self.t_ds.person( p, )
                ds_s.append_person(p, frames_b, pos_vel_b[:,:2], vel=pos_vel_b[:,2:] )

        # save the dataset
        ds_s.save(name)

        return ds_s
        




"""

    Perform measurements on data objects

"""

def get_mean_speed(data, id, roi=((-200, 180),(200, 0)), mode="both", normalize=False, pos=(0, 0)):
    """
        Measures the means spead of a person/ agent in the roi. 

        PARAM:
            data:   data loader object
            frame:  frame id to measure on
            roi:    Measurement region
            mode:   'x'/'y'/'both' select witch velocity to average, both uses the pythagorean velocity
            normalize: normalize with a distane to a referance point.
            pos:    reference position to normalize

        RETURN:
            density:    density in # per are in units of the roi parameters
    """
    frames, pos_vel = data.person(id)
    frames, pos_vel = data.grab_roi(frames, pos_vel, box=roi, x_pad=0 )
    
    if len(frames)==0:
        return np.nan
    
    if not normalize:
        if mode =='x':
             vel_m = pos_vel[:,2].mean()
        elif mode == 'y':
             vel_m = pos_vel[:,3].mean()
        elif mode == 'both':
             vel_m = np.sqrt((pos_vel[:,2:]**2).sum(axis=1)).mean()
    else:
        if mode =='x':
            vel = pos_vel[:,2]
            
            r = pos_vel[0] - pos[0]
            vel /= r
            
            vel_m = vel.mean()
        elif mode == 'y':
            vel = pos_vel[:,3]
            
            r = pos_vel[1] - pos[1]
            vel /= r
            
            vel_m = vel.mean()
        elif mode == 'both':
            vel = np.sqrt((pos_vel[:,2:]**2).sum(axis=1))
            
            r = np.sqrt(( (pos_vel[:,:2]-pos)**2).sum(axis=1))
            vel /= r
            
            vel_m = vel.mean()

    #print("ID ", id ," vel :",  vel_m)
    return vel_m


def get_density(data, frame, roi=((-200, 180),(200, 0)) ):
    """
        Measures the density of People in teh given roi.

        PARAM:
            data:   data loader object
            frame:  frame id to measure on
            roi:    Measurement region

        RETURN:
            density:    density in # per are in units of the roi parameters
    """
    id_s, pos_vel = data.frame(frame)
    
    id, _ = data.grab_roi( id_s, pos_vel, box=roi, x_pad=0, y_pad=0, ret_mask=False )
    
    area = (roi[1][0]-roi[0][0]) * (roi[0][1]-roi[0][0])
    
    density = len(id)/ area
    
    return density


