import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import numpy as np
import pandas
import pickle as pkl
import os

#Parameters describing the limits of the field
xmin=-610     #lower bound of x-scale
xmax=810      #upper bound of x-scale
ymin=-210     #lower bound of y-scale
ymax=410      #upper bound of y-scale
x_steps=141   #absolute resolution on x-axis
y_steps=62    #absolute resolution on x-axis


'''
This script assumes the existence of a folder (in this case the folder 'rawdata')
containing a csv file named sim_name+".csv" containing the data in the format of the 
Data-Loader and a folder named "giffify"+suffix. The folder "giffify"+suffix contains 
three folders "densities" "speeds" and "flows". If the folder does not exist it is created. 

The script will save the following files:
    - flowchart.png in the main folder. Flowchart of all flow data points
      scattered over all densities for all positions in all frames. 
    - velchart.png in the main folder. Chart of all mean speed data points
      scattered over all densities for all positions in all frames. 
    - speed_'frame'.png in giffify/speeds. Heatmap of mean speed at frame 'frame'
      for every frame in range(lowest occuring frame, highest occuring frame,8)
    - density_'frame'.png in giffify/speeds. Heatmap of density at frame 'frame'
      for every frame in range(lowest occuring frame, highest occuring frame,8)
    - flow_'frame'.png in giffify/speeds. Heatmap of flow at frame 'frame'
      for every frame in range(lowest occuring frame, highest occuring frame,8)
    - densities.pkl in main folder. pickle file containing a python list containing
      all occuring densities over all frames and positions
    - flows.pkl in main folder. pickle file containing a python list containing
      all occuring flow values over all frames and positions
'''

"""
    Set here the parameters.
"""

#path of the working folder containing data.csv and giffify
folder_path='data/Output/nn_scan/'
suffix = '_nn1'

sim_name = 'model_nn1'



#load the dataset data.csv using pandas
data=pandas.read_csv(folder_path+sim_name+'.csv',skiprows=1,names=['row','person','frame','y','x','z','vx','vy'])


# Returns the number of people present at frame 'frame'
# in a rcount-radius around the position (x,y)
def density (x,y,frame,data=data,rcount=70):
    idxs=np.array(data['frame']==frame) #find indices of people present in the frame
    dat=np.array(data)[idxs][:,3:5]     #extract their x and y coordinates
    closers=np.linalg.norm(dat-np.array([y,x]),axis=1)<rcount #True for a person closer than rcount to (x,y), else False
    return np.sum(1*closers) #count the number of Trues using 1*True=1 and 1*False=0

# Returns the mean speed of people present at frame 'frame'
# in a rcount-radius around the position (x,y).
def meanspeed (x,y,frame,data=data,rcount=70):
    idxs=np.array(data['frame']==frame) #find indices of people present in the frame
    dat=np.array(data)[idxs][:,3:5]     #extract their x and y coordinates
    closers=np.linalg.norm(dat-np.array([y,x]),axis=1)<rcount #True for a person closer than rcount to (x,y), else False
    if np.sum(closers)==0: #if there are no people present return meanspeed=0
        return 0
    dat=np.array(data)[idxs][:,6:][closers] #get the velocities of the people close to (x,y)
    return np.sum(np.linalg.norm(dat,axis=1))/np.sum(closers*1.0) #return sum of all speeds/number of people in the radius

#This is a useless function, do not use it :)
def surface3d(f,data=data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    # plot a 3D surface like in the example mplot3d/surface3d_demo
    X = np.linspace(xmin, xmax, x_steps)
    Y = np.linspace(ymin, ymax, y_steps)
    x,y = np.meshgrid(X, Y)
    z = np.array([[f(x,y,data) for x in X] for y in Y])
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    ax.view_init(90,30)
    
    #plt.show()
    
# creates heatmap of data
def heatmap(f,data=data,vmin=0,vmax=19,save='',title='',isarray=False,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,x_steps=x_steps,y_steps=y_steps):
    # if isarray is False:
    #    f      : function to be mapped, assumes f(x,y,data) is a number
    #    data   : data given as a parameter to f
    #    vmin   : Lowest value of f that will occur over all frames, Temperature 0 in heatmap
    #    vmin   : Highest value of f that will occur over all frames, max Temperature in heatmap
    #    save   : path where the figure is saved as a png. Not saved if none given.
    #    title  : title to put in the heatmap
    #    xmin,...,   : parameters of the x/y axis, if none given take values given above
    # if isarray is True: THIS CASE HAS NOT YET BEEN TESTED AND IS NEVER USED. It's mainly for shits and giggles
    #    f      : numpy array containing data to be mapped. Assumes f.size=(x_steps,y_steps)
    #    data   : not needed
    #    vmin   : Lowest value in f that will occur over all frames, Temperature 0 in heatmap
    #    vmin   : Highest value of f that will occur, max Temperature in heatmap
    #    save   : path where the figure is saved as a png. Not saved if none given.
    #    title  : title to put in the heatmap
    #    xmin,...,   : parameters of the x/y axis, if none given take values given above
    if isarray:
        fig, ax = plt.subplots()
        X = np.linspace(xmin, xmax, x_steps)
        Y = np.linspace(ymin, ymax, y_steps)
        x,y = np.meshgrid(X, Y)
        z = f
    
        im = ax.imshow(z,cmap='inferno',vmin=0,vmax=vmax)
        ax.figure.colorbar(im, ax=ax)
        if not save=='':
            try:
                plt.savefig(save)
            except:
                None
        if not title=='':
            plt.title(title)
            
        #plt.show()
        return
    fig, ax = plt.subplots()
    X = np.linspace(xmin, xmax, x_steps)
    Y = np.linspace(ymin, ymax, y_steps)
    x,y = np.meshgrid(X, Y)
    z = np.array([[f(x,y,data) for x in X] for y in Y])

    im = ax.imshow(z,cmap='inferno',vmin=vmin,vmax=vmax)
    ax.figure.colorbar(im, ax=ax)
    if not save=='':
        try:
            plt.savefig(save)
        except:
            None
    if not title=='':
        plt.title(title)
        
    #plt.show()
    return
    


# for frame in np.arange(a,b+1,8):
#     print(frame)
#     for x in np.linspace(xmin,xmax,x_steps):
#         for y in np.linspace(ymin,ymax,y_steps):
#             d=density(x,y,frame,rcount=70)
#             if d!=0:
#                 v=meanspeed(x,y,frame,rcount=70)
#                 densities.append(d)
#                 flows.append(v*d)
#     heatmap(lambda x,y,data:density(x,y,frame,rcount=70),save=folder_path+'giffify/densities/density_'+str(frame)+'.png',vmax=19,title="Density")
#     heatmap(lambda x,y,data:meanspeed(x,y,frame,rcount=70),save=folder_path+'giffify/speeds/speed_'+str(frame)+'.png',vmax=242,title="Mean Speed")
#     heatmap(lambda x,y,data:meanspeed(x,y,frame,rcount=70)*density(x,y,frame,rcount=70),save=folder_path+'giffify/flows/flow_'+str(frame)+'.png',vmax=1200,title="Flow")




if not os.path.isdir(folder_path+'giffify'+suffix):
    os.makedirs(os.path.dirname(folder_path+'giffify'+suffix+'/densities/'))
    os.makedirs(os.path.dirname(folder_path+'giffify'+suffix+'/speeds/'))
    os.makedirs(os.path.dirname(folder_path+'giffify'+suffix+'/flows/'))

densities=[]
flows=[]
a=np.min(data['frame'])
b=np.max(data['frame'])
b=100
print('Frames from '+str(a)+' to '+str(b)+':')
for frame in np.arange(a,b+1,8):
    print(frame)
    for x in np.linspace(xmin,xmax,x_steps):
        for y in np.linspace(ymin,ymax,y_steps):
            d=density(x,y,frame,rcount=70)
            if d!=0:
                v=meanspeed(x,y,frame,rcount=70)
                densities.append(d)
                flows.append(v*d)
    heatmap(lambda x,y,data:density(x,y,frame,rcount=70),save=folder_path+'giffify'+suffix+'/densities/density_'+str(frame)+'.png',vmax=19,title="Density")
    heatmap(lambda x,y,data:meanspeed(x,y,frame,rcount=70),save=folder_path+'giffify'+suffix+'/speeds/speed_'+str(frame)+'.png',vmax=242,title="Mean Speed")
    heatmap(lambda x,y,data:meanspeed(x,y,frame,rcount=70)*density(x,y,frame,rcount=70),save=folder_path+'giffify'+suffix+'/flows/flow_'+str(frame)+'.png',vmax=1200,title="Flow")



a_file = open(folder_path+'giffify'+suffix+"/densities.pkl", "wb")
pkl.dump(densities,a_file)
a_file.close()
a_file = open(folder_path+'giffify'+suffix+"/flows.pkl", "wb")
pkl.dump(flows,a_file)
a_file.close()

plt.scatter(densities, flows)
plt.title('Flow Chart')
plt.xlabel('Density')
plt.ylabel('Flow')
plt.savefig(folder_path+'giffify'+suffix+'/flowchart.png')
plt.show()

plt.scatter(np.array(densities),np.array(flows)/np.array(densities))
plt.title('Velocity over Density')
plt.xlabel('Density')
plt.ylabel('velocity')
plt.savefig(folder_path+'giffify'+suffix+'/velchart.png')
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
