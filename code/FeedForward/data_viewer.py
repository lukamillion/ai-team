"""
This module contains plotting functionality for the "AI in pedestrian dynamics" project. The code can be used to visualize pedestrian 
trajectories in various ways.

This code uses the DataLoader object from dataloader.py for input data.

The following functionality is included:

    - plotTraj:         plot the pedestrian trajectories as lines 

    - animateLoc:       Animate the trajectories of pedestrians as a .mp4 or .gif. Here, the position of a pedestrian
                        is indicated by a red/black dot

    - animateTraj:      Animate the trajectories of pedestrians as a .mp4 or .gif. Here, the trajectory of a pedestrian
                        is animated as a continuously appending line plot.

    - animatePreview:   fast and easy preview of DataLoader data to examine simulated data quickly.

    - smooth:           simple running average filter that can be used for various plots. In our case, it is mainly used
                        for training loss plots.

following additional software is needed for the function animatePreview to work:

    - plotly (can be installed via "pip install plotly" in CMD or "conda install plotly" if Anaconda is used)
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='html5')



#plot trajectories
def plotTraj(loader, boundaries, people=None, ai=None, legend=False, wall=False, cor=False, title="Trajectories", path="trajectories.png", save=False):
    """ plot Trajectories with the DataLoader object. People (from dataset) are plotted with red lines, ai_agents are plotted in black.

        PARAM

            loader:         loader object that stores the data
            boundaries:     figure boundaries [xmin, xmax, ymin, ymax]
            people:         List of indexes of the trajectories (people) to plot. If None, prints all
            ai:             List of indexes of that are ai_agents (for color differentiating). If None, no ai_agents are plotted
            legend:         True/False. If True, show figure legend
            wall:           True/False. If True, plots the wall geometry for the ao-360-*** bottleneck
            cor.            True/False, If True, plots the wall geometry for the ug-180-*** corridor
            title:          String. Figure title
            path:           filepath + filename for saving (relative path)
            save:           True/False. If True, save plot
        
        RETURN
            -

        """
    fig = plt.figure(figsize = (10,6))
    # creating a subplot 
    ax1 = fig.add_subplot(1,1,1)
    
    # read people from dataloader if not specified
    if people is None:
        people = loader.data['p'].unique()

    # plot line for every person/agent
    if ai is None:
        for person in people:
            _, traj = loader.person(person)
            ax1.plot(traj[:, 0], traj[:, 1], lw=2, label="{}".format(person))

    else:
        people = set(people) - set(ai)
        for person in people:
            _, traj = loader.person(person)
            ax1.plot(traj[:, 0], traj[:, 1], lw=2, color="red", label="{}".format(person))
        for agent in ai:
            _, traj = loader.person(agent)
            ax1.plot(traj[:, 0], traj[:, 1], lw=2, color="black", label="{}".format(agent))
        
    # figure specifications
    ax1.set_xlim([boundaries[0], boundaries[1]])
    ax1.set_ylim([boundaries[2], boundaries[3]])

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('x Pos. / cm')
    ax1.set_ylabel('y Pos. / cm ')
    ax1.set_title(title, loc="left")

    # plot geometry features (wall/corridor)
    if wall:
        ax1.vlines(-60, ymin=255, ymax=400, lw=3, color="fuchsia")
        ax1.vlines(-60, ymin=-200, ymax=-95, lw=3, color="fuchsia")
    
    if cor:
        # measurement region
        #ax1.vlines(-200, ymin=0, ymax=180, lw=2, color="orange")
        #ax1.vlines(200, ymin=0, ymax=180, lw=2, color="orange")

        # Walls
        ax1.hlines(0, xmin=-300, xmax=300, lw=2, color="fuchsia")
        ax1.hlines(180, xmin=-300, xmax=300, lw=2, color="fuchsia")

    if legend:
        plt.legend()
    
    if save:
        plt.savefig(path, bbox_inches="tight")
        
    plt.show()

    

#Location Animation
def animateLoc(loader, frame_start, frame_stop, boundaries, ai = None, path="loc_anim.gif", save=False, step=1, fps=16, wall=False, cor=False, title="Location Animation", useFFMPEG=False):
    """ Animate Position with the DataLoader object as dots. People (from dataset) are plotted with red dots, 
        ai_agents are plotted in black.

        PARAM

            loader:         loader object that stores the data
            frame_start:    Integer. Start frame number for the animation
            frame_stop:     Integer. Stop frame number for the animation
            boundaries:     figure boundaries [xmin, xmax, ymin, ymax]
            ai:             List of indexes of that are ai_agents (for color differentiating). If None, no ai_agents are plotted
            path:           filepath + filename for saving (relative path)
            save:           True/False. If True, save plot
            step:           frame steps for animation (f.e. step=5 uses every 5th frame)
            fps:            frames per second for the animation. Default is 16 (= actual data fps)
            wall:           True/False. If True, plots the wall geometry for the ao-360-*** bottleneck
            cor.            True/False, If True, plots the wall geometry for the ug-180-*** corridor
            title:          String. Figure title
            useFFMPEG:      True/False. If True, use FFMPEG writer to create videos (must be installed!). 
                            Uses PillowWriter as default.
        
        RETURN
            -

        """
    #preprocess data
    data = []
    ai_data = []

    # load data into new data structure for animation
    for i in np.arange(frame_start, frame_stop, step):
        people, temp = loader.frame(i, ret_vel=False, with_id=False)
        data.append(temp)
        ai_data.append(temp[np.isin(people, ai)])

        
    #Set the figure for the animation framework
    fig = plt.figure(figsize = (10,6)) 
    ax1 = fig.add_subplot(1,1,1)


    # geometry for different datasets
    if wall:
        ax1.vlines(-60, ymin=255, ymax=400, lw=3, color="fuchsia")
        ax1.vlines(-60, ymin=-200, ymax=-95, lw=3, color="fuchsia")

    if cor:
        # measurement region
        ax1.vlines(-200, ymin=0, ymax=180, lw=2, color="orange")
        ax1.vlines(200, ymin=0, ymax=180, lw=2, color="orange")

        # Walls
        ax1.hlines(0, xmin=-300, xmax=300, lw=2, color="fuchsia")
        ax1.hlines(180, xmin=-300, xmax=300, lw=2, color="fuchsia")
    
    scat = ax1.scatter([], [], c="red")
    scat_ai = ax1.scatter([], [], c="black")
    ax1.set_xlim([boundaries[0], boundaries[1]])
    ax1.set_ylim([boundaries[2], boundaries[3]])

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('x Pos. / cm')
    ax1.set_ylabel('y Pos. / cm ')
    ax1.set_title(title, loc="left")

    # animation function that is called for each frame
    def animate(i):
        scat.set_offsets(data[i])
        scat_ai.set_offsets(ai_data[i])
        return scat,

    frames = int(np.floor((frame_stop - frame_start)/step))
    
    ani = animation.FuncAnimation(fig = fig, func = animate, frames =frames, interval = int(step*1000/fps), blit=True)
    plt.close(fig)
    
    # save animation to .mp4 or .gif via writer
    if save:
        if useFFMPEG:
            writer = animation.FFMpegWriter(fps=fps/step, extra_args=['-vcodec', 'libx264'])
        else:
            writer = animation.PillowWriter(fps=fps/step, extra_args=['-vcodec', 'libx264'])
        ani.save(path, writer=writer)
    return ani



# Trajectory animation
def animateTraj(loader, frame_start, frame_stop, boundaries, wall=False, cor=False, ai=None, path="traj_anim.gif", save=False, step=1, fps=16, title="Trajectory Animation", useFFMPEG=False):
    """ Animate Trajectory with the DataLoader object as lines. Here, the trajectories are rendered as continuously expanding
        lines. People (from dataset) are plotted with red lines, ai_agents are plotted in black.

        PARAM

            loader:         loader object that stores the data
            frame_start:    Integer. Start frame number for the animation
            frame_stop:     Integer. Stop frame number for the animation
            boundaries:     figure boundaries [xmin, xmax, ymin, ymax]
            ai:             List of indexes of that are ai_agents (for color differentiating). If None, no ai_agents are plotted
            path:           filepath + filename for saving (relative path)
            save:           True/False. If True, save plot
            step:           frame steps for animation (f.e. step=5 uses every 5th frame)
            fps:            frames per second for the animation. Default is 16 (= actual data fps)
            wall:           True/False. If True, plots the wall geometry for the ao-360-*** bottleneck
            cor.            True/False, If True, plots the wall geometry for the ug-180-*** corridor
            title:          String. Figure title
            useFFMPEG:      True/False. If True, use FFMPEG writer to create videos (must be installed!). 
                            Uses PillowWriter as default.
        
        RETURN
            -

        """
    # prepare data for animation
    data = []
    person = []
    colors = []

    p_ = loader.data['p'].unique()
    people_count = int(p_[p_ < 1000].max())
    print(people_count)

    # load data in data structure for animation
    for i in np.arange(frame_start, frame_stop, step):
        data.append(loader.frame(i, ret_vel=False, with_id=False)[1])
        person.append(loader.frame(i, ret_vel=False, with_id=False)[0])

    #Set the figure for the animation framework
    fig = plt.figure(figsize = (10,6))
    #creating a subplot 
    ax1 = fig.add_subplot(1,1,1)

    # figures specds
    ax1.set_xlim([boundaries[0], boundaries[1]])
    ax1.set_ylim([boundaries[2], boundaries[3]])

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('x Pos. / cm')
    ax1.set_ylabel('y Pos. / cm ')
    ax1.set_title(title, loc="left")

    # dataset geometry
    if wall:
        ax1.vlines(-60, ymin=255, ymax=400, lw=3, color="fuchsia")
        ax1.vlines(-60, ymin=-200, ymax=-95, lw=3, color="fuchsia")

    if cor:
        # measurement region
        ax1.vlines(-200, ymin=0, ymax=180, lw=2, color="orange")
        ax1.vlines(200, ymin=0, ymax=180, lw=2, color="orange")

        # Walls
        ax1.hlines(0, xmin=-300, xmax=300, lw=2, color="fuchsia")
        ax1.hlines(180, xmin=-300, xmax=300, lw=2, color="fuchsia")

    #initialize line objects for plotting
    lines = []
    vals = []

    if ai is None:
        for i in range(people_count):
            lobj = ax1.plot([],[], lw=2)[0]
            lines.append(lobj)
            vals.append([[], []])
    else:
        for i in range(people_count):
            if (i+1001) in ai:
                lobj = ax1.plot([],[], color="black", lw=2)[0]
            else:
                lobj = ax1.plot([],[], color="red", lw=2)[0]
            lines.append(lobj)
            vals.append([[], []])

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    #Animation function that is called for each frame
    def animate(i):
    
        #update data for plotting
        for (per, dat) in zip(person[i], data[i]):

            if per > 1000:
                per -= 1000

            vals[int(per-1)][0].append(dat[0])
            vals[int(per-1)][1].append(dat[1])
            
        #write new data to line objects
        for lnum, line in enumerate(lines):
            line.set_data(vals[lnum][0], vals[lnum][1])
        return lines

    frames = int(np.floor((frame_stop - frame_start)/step))
    ani = animation.FuncAnimation(fig = fig, func = animate, frames = frames, interval = int(step*1000/fps), blit=True) 
    plt.close(fig)
    
    # save animation by writing frames to .mp4 or .gif via writer
    if save:
        if useFFMPEG:
            writer = animation.FFMpegWriter(fps=fps/step, extra_args=['-vcodec', 'libx264'])
        else:
            writer = animation.PillowWriter(fps=fps/step, extra_args=['-vcodec', 'libx264'])
        ani.save(path, writer=writer)
    return ani



#fast data preview
def animatePreview(loader, boundaries, step):
    """ quickly preview Animation in a jupyter notebook, using plotly.
    
        PARAM

            loader:         DataLoader object that stores the data
            boundaries:     figure boundaries [xmin, xmax, ymin, ymax]
            step:           frame-step from one image to the next 
                            (step=1 means all frames are plotted, step=5 everey 5th)

        RETURN
            -

    """
    import plotly.express as px
    fig = px.scatter(loader.data[(loader.data['f'] % 10) == 0], 
           x="x", y="y", 
           animation_frame="f", animation_group='p', hover_name="p",
           range_x=[boundaries[0], boundaries[1]], range_y=[boundaries[2], boundaries[3]],
           template="plotly_white", title="Animation Preview")
    fig.show()



# running average filter 
def smooth(y, box_pts):
    """ Smoothen input data y with a running average filter given by the length box_pts.

    PARAM

        y:          Input data to smooth, as numpy array
        box_pts:    Running average window length
        
    RETURN
        y_smooth:   smoothened data with same shape as y

    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
