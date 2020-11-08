import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

"""

    Dataloader utilities designed for the Pedestriandata of the Jülich Forschungszentrum:

    https://ped.fz-juelich.de/da/doku.php?id=start#unidirectional_flow_closed_boundary_condition

    
    UG Dataset:
    Frame rate: 16 FPS
    Data is stored in the following format:
    Person id | frame | x pos | y pos | z pos 


"""





# TODO: define ROI





class DataLoader():
    """
    
    Class to load Pedestrian data of the 
    
    """
    def __init__(self, path, fps=16, FlipX=True, FlipY=False):
        self.path = path
        self.fps = fps
        self.FlipX = FlipX
        self.FlipY = FlipY
        
        with open(path) as f:
            df = df = pd.read_fwf(self.path, infer_nrows=10001, header=None)#, colspecs=colspecs, index_col=0)
            df.columns = ['p', 'f', 'y', 'x', 'z', ] # Chang x and y because date is stored transposed
        
        l = len(df)
        cor = np.vstack((np.ones(l)*(-1)**self.FlipX, np.ones(l)*(-1)**self.FlipY )).T
        
        df[['x', 'y']] = df[['x', 'y']].to_numpy()*cor
        
        
        
        
        self.data = df
        
        dfv = pd.DataFrame(index=df.index, columns=['vx', 'vy'])
        
        for p_id in df.p.unique():
        
            idx, _, pers = self.person(p_id, ret_vel=False, with_id=True)

            vx, vy = self.get_vel_(pers, self.fps)

            dfv['vx'][idx] = vx
            dfv['vy'][idx] = vy
        
        
        self.data = df.join(dfv)
        
        
    def get_vel_(self, pos, fps=-1, ):
        T = 1/ (fps if fps>0 else self.fps)

        vx = np.diff( pos[:,0], append=pos[-1,0] ) / T
        vy = np.diff( pos[:,1], append=pos[-1,1] ) / T
        
        return vx, vy
        
    
    def person(self, id, ret_vel=True, with_id=False):
        self.temp = self.data[self.data['p'] == id]
        
        if ret_vel:
            ret_col = ['x', 'y', 'vx', 'vy' ]
        else: 
            ret_col = ['x', 'y']
        
        if not with_id:
            return self.temp['f'].to_numpy(), self.temp[ret_col].to_numpy()
        else:
            idx = self.temp.index
            return idx, self.temp['f'].to_numpy(), self.temp[ret_col].to_numpy()
    
    
    def frame(self, id, ret_vel=True, with_id=False):
       
        self.temp = self.data[self.data['f'] == id]
        
        if ret_vel:
            ret_col = ['x', 'y', 'vx', 'vy' ]
        else: 
            ret_col = ['x', 'y']
        
        if not with_id:
            return self.temp['p'].to_numpy(), self.temp[ret_col].to_numpy()
        else:
            idx = self.temp.index
            return idx, self.temp['p'].to_numpy(), self.temp[ret_col].to_numpy()
    
    
    def get_nn(self, pids, pos, idx, nn,  fill=True, include_origin=False):
        ref = pos[idx, :2]
        
        dist = ((pos[:,:2]-ref)**2).sum(axis=1)
        sort = np.argsort(dist)
               
        sort = sort[ int(not include_origin):nn+1] # plus one cause first is 
        
        pos = pos[sort]
        
        
        pids = pids[sort]
        
        if fill and pos.shape[0]<nn+include_origin:
            
            
            z = np.zeros((nn+include_origin, pos.shape[1]))
            print(pos.shape)
            print(z.shape)
            z[:pos.shape[0]] = pos
            pos = z
            z = np.zeros(nn+include_origin)
            z[:pids.shape[0]] = pids
            pids = z
         
        return pids, pos
        
    
    def frame_nn(self, f_id, p_id, nn=4, ret_vel=True, fill=True, include_origin=False, with_id=False):
        idx, p_ids, pos_vel = self.frame(f_id, with_id=True)
        
        if not p_id in p_ids:
            raise IndexError(  "Person not present in Frame"  )    
        
        p_ids, pos_vel_nn = self.get_nn(p_ids, pos_vel, np.where(p_ids==p_id)[0], nn, fill=fill, include_origin=include_origin)

        if not with_id:
            return p_ids, pos_vel_nn
        else:
            return idx, p_ids, pos_vel_nn
    


