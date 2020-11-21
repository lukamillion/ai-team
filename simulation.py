import numpy as np
import pandas as pd
import torch
import multiprocessing
import progressbar
from progressbar import FormatLabel, Percentage, Bar, ETA

from data_loader import DataLoader



class Agent():
    def __init__(self, model, pos_vel_0, id=100, frame_0=0, T=16, truth_with_vel=False, device="cpu"):
        
        
        self.id = id
        
        self.model = model
        self.device = device
        self.truth_with_vel = truth_with_vel
        
        if self.device.type.startswith("cuda"):
            self.model.cuda()
        
        #print(pos_vel_0)
        self.pos_vel_0 = pos_vel_0.copy()
        self.frame_c = frame_0
        
        self.T = T
        
        # position and velocity data
        self.frames = [frame_0]
        self.traj = [pos_vel_0]
                
    @property 
    def pos(self):
        return self.traj[-1][ :2]
    
    @property
    def vel(self):
        return self.traj[-1][ 2:]
    
    def run(self, neighbors):
        return self.step(neighbors)
    
    def step(self, neighbors):
        
        with torch.no_grad():
            x_sim = torch.from_numpy(neighbors).to(self.device)

            y_sim = self.model(x_sim.float())
            
        if self.truth_with_vel:
            v_sim = y_sim[2:]
            y_sim = y_sim[:2]
        else:
            v_sim = (y_sim[:2]-x_sim[:2])/self.T
        
        self.traj.append( np.concatenate( (y_sim.cpu().detach().numpy(), v_sim.cpu().detach().numpy()) ) )
        self.frame_c +=1
        self.frames.append(self.frame_c)
        
        return self.traj[-1]
        




class Engine():
    def __init__(self, ds, agents=[], nn=10, stop_agent=False, mode="wraps", exportpath="sim.csv"):
        
        self.ds = DataLoader(exportpath)
        self.ds.copy(ds)
        
        self.agents = agents
        
        self.nn = nn
        self.mode = mode
        self.stop_agent = stop_agent
        
        self.cur_f = 0
        
    def step(self, ):
        
        in_frame, pos_vel_f = self.ds.frame(self.cur_f)
        for a in self.agents:
            if a.pos[0]>500 and self.stop_agent:
                continue
            
            if a.id not in in_frame :
                continue
            #_, pos_vel = self.ds.frame_nn(self.cur_f, a.id, nn=self.nn, use_roi=False, mode=self.mode)
            _, pos_vel, _ = self.ds.get_nn(in_frame, pos_vel_f, np.where(in_frame==a.id)[0],
                                           self.nn,
                                           mode=self.mode   )
            
            n_pos_vel = a.step(pos_vel.copy().ravel()).copy()
            
            
            n_pos_vel [[0, 1]] = n_pos_vel [[1, 0]]
           
            
            # TODO write to ds
            entry = np.concatenate( ([a.id], [self.cur_f+1], n_pos_vel[:2], [0], n_pos_vel[2:] ) )
            
            if len(self.ds.data[(self.ds.data["p"]==a.id) & (self.ds.data["f"]==self.cur_f+1)]):
                self.ds.data[(self.ds.data["p"]==a.id) & (self.ds.data["f"]==self.cur_f+1)] = [entry]
            else:
                self.ds.data = self.ds.data.append(pd.DataFrame([entry],
                                                                columns=list(self.ds.data)),
                                                               ignore_index=True)    
                
            

        self.cur_f += 1
        
    
    def run(self, start_f, stop_f, ):
        self.cur_f = start_f
        print("sim from : {} to {}".format(start_f, stop_f))
        widgets = [FormatLabel(''), ' ', Percentage(), ' ', Bar(), ' ', ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=stop_f)
        pbar.start()
        
        for  a in self.agents:

            pos = a.pos_vel_0[:2]
            
            entry = np.concatenate( ([a.id],  [a.frame_c], np.flip(pos) , [0],  a.pos_vel_0[2:]) )
            
            if len(self.ds.data[(self.ds.data["p"]==a.id) & (self.ds.data["f"]==a.frame_c)]):
                self.ds.data[(self.ds.data["p"]==a.id) & (self.ds.data["f"]==a.frame_c)] = [entry]
            else:
                self.ds.data = self.ds.data.append(pd.DataFrame([entry],
                                                                columns=list(self.ds.data)),
                                                               ignore_index=True)
        
        
        while self.cur_f < stop_f:
            self.step()
            #print("===RUN===")
            #print(self.agents[-1].traj[-1])
            #print(self.ds.data[(self.ds.data["p"]==self.agents[-1].id) & (self.ds.data["f"]==self.cur_f)])
            widgets[0] = FormatLabel('frame: {:4}'.format(self.cur_f))
            pbar.update(self.cur_f)
        
        pbar.finish()


    
    def save(self, ):
        pass