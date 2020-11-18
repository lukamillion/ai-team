import numpy as np
import pandas as pd

import copy


import matplotlib.pyplot as plt


import progressbar
from progressbar import FormatLabel, Percentage, Bar, ETA

"""

    Dataloader utilities designed for the Pedestriandata of the Jülich Forschungszentrum:

    https://ped.fz-juelich.de/da/doku.php?id=start#unidirectional_flow_closed_boundary_condition

    
    UG Dataset:
    Frame rate: 16 FPS
    Data is stored in the following format:
    Person id | frame | x pos | y pos | z pos 


"""



class DataLoader():
    """
    
    Class to load Pedestrian data of the Juelich Froschungszenturm 
    
    """
    def __init__(self, path, fps=16, FlipX=True, FlipY=False):
        """
        
            Load the textfile into the class. The

            Param: 
                path: path tho the textfile
                fps:  frame rate per second of the data
                FlipX: invert axis in X direction
                FlipY: invert axis in Y direction

            Return: 
                -

        """
        self.path = path
        self.fps = fps
        self.FlipX = FlipX
        self.FlipY = FlipY
        

        self.data = pd.DataFrame(columns=['p', 'f', 'x', 'y', 'z', 'vx', 'vy'])
        
    @property
    def frames(self):
        return len(self.data.f.unique())

    @property
    def persons(self):
        return len(self.data.p.unique())
        
    def get_vel_(self, pos, fps=-1, ):
        """
            Computes the velocity for the given positions.

            Param:
                pos: 2D-NP array with time in axis 0 and x, y in axis 1
                fps: frame rate default -1 use fps of dataset
            Return:
                vx, vy: NP array with all the velocities
        """
        T = 1/ (fps if fps>0 else self.fps)

        vx = np.diff( pos[:,0], append=pos[-1,0] ) / T
        vy = np.diff( pos[:,1], append=pos[-1,1] ) / T
        
        return vx, vy
        
    
    def person(self, id, ret_vel=True, with_id=False):
        """
            Returns the trajectory of a person.

            Param:
                id: id number of person
                ret_vel: returns x,y velocities
                with_id: returns row id in the dataset

            Return:
                idx: if with_id is true return unique row_id
                frame: return frame count
                pos_vel: returns [x, y, (vx, vy)] as 2d array with time in axis 0 
        """
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
        """
            Returns all persons in a given fram

            Param:
                id: id of frame
                ret_vel: returns x,y velocities
                with_id: returns row id in the dataset

            Return:
                idx: if with_id is true return unique row_id
                p_id: return person id
                pos_vel: returns [x, y, (vx, vy)] as 2d array with time in axis 0 
        """
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
    
    
    def get_nn(self, pids, pos, idx, nn,  fill=True, mode="zero", include_origin=True):
        """
            Get the N nearest neighbors to the given position array.

            Param:
                pids: Person id's
                pos:  2D array with position and ev. velocities in formtat [[x, y, (vx, vy)]]
                idx:  0 index of the reference positon this is not the person id
                nn: number of neighbors 
                fill: fill array with zeros if not enough neighbors
                mode: set fill behavior
                        zero: fills with zeros 
                        wrap: fills with existing neighbors or self if no neighbors
                        number: fills with the given number
                include_origin: if true first entry is the origin person

            Return:
                idx: if with_id is true return unique row_id
                p_ids: id of Nearest neighbors order from near to factor length nn
                p_pos_Vel: Nearest neighbors pos and velocities  [[x, y,(vx,vy)],  ] shape(nn, 2(4))
        """
        ref = pos[idx, :2]
        
        dist = ((pos[:,:2]-ref)**2).sum(axis=1)
        sort = np.argsort(dist)
               
        sort = sort[ int(not include_origin):nn+1] # plus one cause first is 
        
        pids = pids[sort]
        pos = pos[sort]
        

        filled = False

        if fill and pos.shape[0]<nn+include_origin:
            
            filled = True

            if mode=="wrap" and pos.shape[0]>include_origin:
                neig_id = pids[int(include_origin):]
                neig = pos[int(include_origin):]

                m= int(np.ceil(nn/neig.shape[0] ))
                neig_id  = np.tile(neig_id, m)
                neig = np.tile(neig.T, m).T

                if include_origin:
                    pids = np.insert(neig_id, 0, pids[0], axis=0)
                    pos = np.vstack((pos[0], neig))
                
                pids = pids[:nn+include_origin]
                pos = pos[:nn+include_origin]

            else:
                if mode=="zero" or mode=="wrap":        # if no neighbor is there we use zero instead
                    k = 0 
                
                else:
                    k = mode
                
                z = np.ones((nn+include_origin, pos.shape[1]))*k

                z[:pos.shape[0]] = pos
                pos = z
                z = np.ones(nn+include_origin) * k
                z[:pids.shape[0]] = pids
                pids = z
        
        if fill:
            return pids, pos, filled
        else:
            return pids, pos
        
    
    def frame_nn(self, f_id, p_id, nn=4, ret_vel=True, fill=True, mode="zero", use_roi=True, include_origin=True, ret_full=False, box=((-300, 100), (300,0)), x_pad=50, y_pad=0):
        """
            Get the N nearest neighbors to the given person id and frame.

            Param:
                f_id: Frame id
                p_id: person id
                nn: number of neighbors 
                ret_vel: if true returns velocities
                fill: fill array with zeros if not enough neighbors
                mode: set fill behavior
                        zero: fills with zeros 
                        wrap: fills with existing neighbors or self if no neighbors
                        number: fills with the given number
                include_origin: if true first entry is the origin person
                with_id: if ture returns unique row ids

            Return:
                p_ids: id of Nearest neighbors order from near to factor length nn
                p_pos_Vel: Nearest neighbors pos and velocities  [[x, y,(vx,vy)],  ] shape(nn, 2(4))
        """
        idx, p_ids, pos_vel = self.frame(f_id, with_id=True)
        
        if not p_id in p_ids:
            raise IndexError("Person {} not present in Frame {}".format(p_id, f_id))    
        
        if use_roi:
            p_ids, pos_vel, mask = self.grab_roi(p_ids, pos_vel, box=box, x_pad=x_pad, y_pad=y_pad, ret_mask=True    )
            idx = idx[mask]
            
            if not p_id in p_ids:
                raise IndexError("Person {} not present in selected ROI (frame {})".format(p_id,   f_id )) 

        filled = False
        if fill:
            p_ids, pos_vel_nn, filled = self.get_nn(p_ids, pos_vel, np.where(p_ids==p_id)[0], nn, fill=fill, mode=mode,  include_origin=include_origin)
        else:
            p_ids, pos_vel_nn= self.get_nn(p_ids, pos_vel, np.where(p_ids==p_id)[0], nn, fill=fill, mode=mode,  include_origin=include_origin)

        
        if not ret_vel:
            pos_vel_nn = pos_vel_nn[:,0:2]

        if not ret_full:
            return p_ids, pos_vel_nn.astype(np.float)
        else:
            return idx, p_ids, pos_vel_nn.astype(np.float), filled


    def grab_roi(self, id_s, pos_vel, box=((-300, 100), (300,0)), x_pad=50, y_pad=0, ret_mask=False ):
        # pre, post length
        # square
        # 
        """
            
            Only returns the positions witch both coordinates in the 
            roi region. 

            Param: 
                ID_s: Person id to check
                pos_vel: postiontions to check in fromat [[x, y, ....]...]
                box: ((x_ul, yul),(xlr, ylr))
                x_pad: padding in x direction
                y_pad: padding in y direction

            Return:
                Id_s:   IDs of the persons in the roi
                Pos_vel: Positions an velocities in the roi
            
        """
        x_low, x_high = box[0][0]-x_pad, box[1][0]+x_pad
        y_low, y_high = box[1][1]+y_pad, box[0][1]-y_pad

        m1 = np.ma.masked_where( pos_vel[:,0]<x_high, pos_vel[:,0]).mask
        m2 = np.ma.masked_where( pos_vel[:,0]>x_low, pos_vel[:,0]).mask
        m3 = np.ma.masked_where( pos_vel[:,1]<y_high, pos_vel[:,1]).mask
        m4 = np.ma.masked_where( pos_vel[:,1]>y_low, pos_vel[:,1]).mask
        
        mask = m1*m2*m3*m4

        if not ret_mask:
            return id_s[mask], pos_vel[mask]
        else:
            return id_s[mask], pos_vel[mask], mask


    def get_trajectories(self, nn, ret_vel=True, nn_vel=True, fill=True, mode="zero", omit_no_neighbors=False, use_roi=False,  box=((-300, 100), (300,0)), x_pad=50, y_pad=0,):
        """
            Return A list with all trajectories through the corridor.

            Param:
                nn: number of nearest neighbors
                ret_vel: if true uses velocities
                fill: keep constant size of neighbors
                mode: set fill behavior
                        zero: fills with zeros 
                        wrap: fills with existing neighbors or self if no neighbors
                        number: fills with the given number
                omit_no_neighbors: do not use trajectories which do not have enough neighbors
                box: ROI box
                x_pad/ y_pad: padding of the box

            Return:
                train_id: person id's which correspond to the trajectories
                trajectories: trajectories with neighbors

        """

        # return list of singele paths with nn neighbors
        # and corresponding ids
        train_id = []
        trajectories = []

        pbar = progressbar.ProgressBar(maxval=self.persons)
        pbar.start()

        for id_p in range(1, 1+self.persons):
            frames, pos_vel = self.person(id_p)      

            if use_roi:  
                roi_f, roi_p = self.grab_roi(frames, pos_vel, box, x_pad, y_pad, )
            else:
                roi_f, roi_p = frames, pos_vel

            filled = 0

            traj = []

            for f in roi_f:
                _, _, pos_neig, np_f = self.frame_nn(f, id_p, nn, ret_vel=ret_vel, fill=fill, mode=mode, use_roi=use_roi, box=box, x_pad=x_pad, y_pad=y_pad, ret_full=True)
                filled += np_f
                
                if ((not nn_vel) & ret_vel):
                    pos_neig = np.concatenate((pos_neig[0,:], pos_neig[1:,0:2].ravel()))

                traj.append( pos_neig.ravel())

            if len(traj)==0:
                continue

            traj = np.array(traj)

            if filled==0 or not omit_no_neighbors:
                train_id.append(id_p)
                trajectories.append(traj.astype(np.float32))
               
            pbar.update(id_p)

        pbar.finish()

        return train_id, trajectories


    def flip_x(self, traj, x_center=0, use_vel=True):
        """
            Flip the trajectories around the mirror axis at x_center.

            Param: 
                traj: Data to process
                x_center: center oft the mirror axis 
                use_vel: if true the data is encoding velocity

            return: 
                traj: traj and flipped traj length increases by a factor of 2

        """

        flip = [-1, 0]
        flip = np.tile(flip, int(traj[0].shape[1]/2))

        if use_vel:
            center = [1, 0, 0, 0]
        else:
            center = [1, 0 ]
        center = np.tile(center, int(traj[0].shape[1]/len(center))) * x_center

        aug = []

        for t in traj:
            steps_aug = t-center
            steps_aug *= flip
            steps_aug = steps_aug+center
            aug.append(steps_aug)


        return traj+aug


    def batch_(self, arr, n):
        """ yiels successive n-sized batches from arr """
        out = []
        for i in range(len(arr)-1-n): 
            out.append(arr[i:i+n].ravel())

        return np.vstack(out)



    def trajectory_2_steps(self, trajectories, step_nr=1, truth_with_vel=False):
        """
            
            Stacks multiple trajectories on top of each other
            we loose the information of of a trajectory and can only look at the next steps

            Param:
                trajectories: List of multiple trajectories
                step_nr: number of steps given as input for each person
                truth_with_vel: if true velocity is encoded in the data 

            return: 
                input_steps: array of steps at boarder of trajectories there are jumps
                truth_steps: the truth date for each step



        """

        inputs = []
        truths = []

        if step_nr == 1:
            for t in trajectories:
                inputs.append(t[:-1])
                truths.append(t[1:, :2+2*truth_with_vel])

            return np.vstack(inputs), np.vstack(truths)

        else:
            for t in trajectories:
                inputs.append(self.batch_(t, step_nr))
                truths.append(t[1:, :2+2*truth_with_vel])

            return np.vstack(inputs), np.vstack(truths)



    def get_train_data(self, nn, step_nr=1, augmentation=[], truth_with_vel=False, split=(60, 20, 20), shuffle=True, **kwargs):
        """
            
            Get train, validation and test data from the dataset. 

            Param:
                nn: number of neighbors
                Augmentation: a list of augmentation functions that work on a list of trajectories
                truth_with_vel: if true the velocity is stored in the truth data 
                split: how to split the data
                shuffle: if true all the data is shuffled 
                **kwargs: pass on tho the get_trajectories method 
            return: 
                (train_in, trian_truth): training data with corresponding truth values
                (val_in, val_truth): validation data with corresponding truth values
                (test_in, test_truth): testing data with corresponding truth values

        """
        idexs, trajs = self.get_trajectories(nn=nn, **kwargs)
        print("loaded {} trajectories".format(len(idexs)))


        for aug in augmentation:
            trajs = aug(trajs)

        print("with augmentation {} trajectories".format(len(trajs)))

        steps_input, steps_truth = self.trajectory_2_steps(trajs, step_nr, truth_with_vel)

        if shuffle:
            p = np.random.permutation(len(steps_truth))
            #np.random.shuffle(steps_input)
            steps_input = steps_input[p]
            steps_truth = steps_truth[p]

        print("extracted {} steps".format(len(steps_input)))

        

        length = len(steps_input)
        train, test = int(split[0]/100.0*length), int(split[2]/100.0*length)
        print(train, test)

        train_in = steps_input[:train].astype(np.float32)
        train_truth = steps_truth[:train].astype(np.float32)

        val_in = steps_input[train:(length - test)].astype(np.float32)
        val_truth = steps_truth[train:(length - test)].astype(np.float32)

        test_in = steps_input[(length - test):].astype(np.float32)
        test_truth = steps_truth[(length - test):].astype(np.float32)

        return (train_in, train_truth), (val_in, val_truth), (test_in, test_truth)
    

    def graph_traj(self, id, traj):
        """
            graph a trajectory 

            Param:
                id: id of person to graph 
                traj: array with all the neighbors traj

            Return:
                -
        """
        frames, pos_vel = self.person(id)
        nn = int(traj.shape[1]/4)

        plt.figure()
        plt.scatter(pos_vel[::10,0],pos_vel[::10,1], c="orange", label="no roi person")

        plt.scatter(traj[::10,0],traj[::10,1], c="b", label="person in roi")
        if traj.shape[0]>0:
            for j in range(nn):
                plt.plot(traj[:,0+j*4], traj[:,1+j*4], label="neig {}".format(j))

        plt.plot([-300, 300], [100, 100], c="k", lw="5", label="wall")
        plt.plot([-300, 300], [0, 0], c="k", lw="5")
        plt.legend(loc = 4)
        plt.show()

    def remove_person(self, id):
        """
            Removes the person with the given id from the dataset. 

            Param:
                id: person id to remove 
            Return:
                -
        """
        self.data.drop(self.data[self.data.p==id].index, inplace=True)
        

    def replace_person(self, id, frame, traj, vel=None):
        """
            Replaces the trajectory of the person with the ID 

            Param:
                id: person id to replace 
                frame: array of frames of the trajectory
                traj: array of shape [[ x, y], ... ] encoding the positions
            Return:
                - 
        """
        self.remove_person(id)
        self.append_person(id, frame, traj, vel=vel)
        

    def append_person(self, id, frame, traj, vel=None):
        """
            Append at trajectory for a new person ID.

            Param:
                id: person id to  append this must not be in the dataset
                    convetion use id>100 for AI agents
                frame: array of frames of the trajectory
                traj: array of shape [[ x, y], ... ]encoding the positions
                vel: optional array of velocities to append shape [[ vx, vy], ... ]
            Return:
                -
        """

        if id in self.data.p.to_list():
            raise IndexError("Person ID already exists.")

        l = traj.shape[0]

        if vel is None:
            vel = np.empty((l, 2))


        data = np.hstack((np.ones((l,1))*id, frame.reshape((l,1)), traj, np.zeros((l,1)), vel) )
        self.data = self.data.append( pd.DataFrame( data, columns=['p', 'f', 'x', 'y', 'z', 'vx', 'vy']),  ignore_index=True )


    def copy(self, data):
        """
            Makes a deepcopy of the data and settings of the given data

            Param:
                data: object to copy from, path is not copied!

            Return:
                -
        """
        self.fps = data.fps
        self.FlipX = data.FlipX
        self.FlipY = data.FlipY

        self.data = copy.copy(data.data)

    def load(self,):
        """
            load data from the given path

        """
        if self.path[-3:] == "txt":
            with open(self.path) as f:
                file_length = len(f.readlines(  ))
                df = pd.read_fwf(self.path, infer_nrows=file_length, header=None)#, colspecs=colspecs, index_col=0)
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

            print("loaded {} persons".format(self.persons))

        elif self.path[-3:]=="hd5":
            self.data = pd.read_hdf(self.path, name="Dataset", mode="r")
        elif self.path[-3:]=="csv":
            self.data = pd.read_csv(self.path,index_col=0)
        else:
            print("unknown dataformat!!")
        

    def save(self, path, as_hd5=False ):
        """
            Saves the data as CSV or hd5 file.

            Param:
                path: path and name of file the ending is appended 
                as_hd5: True file is saved as hd5 file
                        False file is stored as csv
        """
        if as_hd5:
            self.data.to_hdf(path+".hd5", key="Dataset", mode='w')
        else:
            self.data.to_csv(path+".csv" )



