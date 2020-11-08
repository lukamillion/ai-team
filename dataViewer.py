import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='html5')

"""
following additional software is needed for this code to work:

    - for saving: FFmpeg (https://ffmpeg.org/download.html)
    - for animatePreview: plotly

TODO: Implement additional data input for AI-simulated trajectories / comparision trajectories
-> fix data format first!
"""

#plot trajectories
def plotTraj(loader, boundaries, people=None, ai=None, legend=False, title="Trajectories", path="trajectories.png", save=False):
    """ plot Trajectories with the loader object.

        loader:     loader objekct that stores the data
        boundaries: figure boundaries [xmin, xmax, ymin, ymax]
        people:     trajectories (people) to plot. If None, prints all
        unicolor:   if True, print all trajecotires from people in Red
        ai:         people number that are ai_agents (for color grading)
        legend:     show figure legend
        title:      figure Title
        path:       filepath + filenahme for saving
        save:       if True, save plot

        """
    fig = plt.figure(figsize = (10,6))
    #creating a subplot 
    ax1 = fig.add_subplot(1,1,1)
    
    if people is None:
        people = np.arange(loader.data['p'].min(), loader.data['p'].max())


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
        
    ax1.set_xlim([boundaries[0], boundaries[1]])
    ax1.set_ylim([boundaries[2], boundaries[3]])

    ax1.set_xlabel('x Pos. / cm')
    ax1.set_ylabel('y Pox. / cm ')
    ax1.set_title(title, loc="left")
    
    if legend:
        plt.legend()
    
    if save:
        plt.savefig(path, bbox_inches="tight")
        
    plt.show()
    
#Location Animation
def animateLoc(loader, frame_start, frame_stop, boundaries, ai = None, path="loc_anim.mp4", save=False, step=1, fps=16, title="Location Animation"):
    """ Animate the Trajectory as lines

        loader:     loader objekct that stores the data
        frame_start:frame where the animation starts
        frame_stop: frame where the animation stops
        boundaries: figure boundaries [xmin, xmax, ymin, ymax]
        ai:         person list to select the ai agents for color grading
        path:       filepath + filenahme for saving
        save:       if True, save plot
        step:       frame steps for animation (f.e. step=5 uses every 5th frame)
        fps:        frames per second for the animation (with step=1)
        title:      Title of the Animation
    """
    #preprocess data
    data = []
    ai_data = []

    for i in np.arange(frame_start, frame_stop, step):
        people, temp = loader.frame(i)
        data.append(temp)
        ai_data.append(temp[np.isin(people, ai)])
        
    #Set the figure for the animation framework
    fig = plt.figure(figsize = (10,6))
    #creating a subplot 
    ax1 = fig.add_subplot(1,1,1)
    
    scat = ax1.scatter([], [], c="red")
    scat_ai = ax1.scatter([], [], c="black")
    ax1.set_xlim([boundaries[0], boundaries[1]])
    ax1.set_ylim([boundaries[2], boundaries[3]])

    ax1.set_xlabel('x Pos. / cm')
    ax1.set_ylabel('y Pox. / cm ')
    ax1.set_title(title, loc="left")

    #Using FuncAnimation we need to create an animation function which return and/or done a repetitive action
    def animate(i):
        scat.set_offsets(data[i])
        scat_ai.set_offsets(ai_data[i])
        return scat,

    frames = int(np.floor((frame_stop - frame_start)/step))
    
    ani = animation.FuncAnimation(fig = fig, func = animate, frames =frames, interval = int(step*1000/fps), blit=True)
    plt.close(fig)
    
    if save:
        ani.save(path, fps=fps, extra_args=['-vcodec', 'libx264'])
    return ani



# Trajectory animation
def animateTraj(loader, frame_start, frame_stop, boundaries, ai=None, path="traj_anim.mp4", save=False, step=1, fps=16, title="Trajectory Animation"):
    """ Animate the Trajectory as lines

        loader:     loader objekct that stores the data
        frame_start:frame where the animation starts
        frame_stop: frame where the animation stops
        boundaries: figure boundaries [xmin, xmax, ymin, ymax]
        ai:         person list to select the ai agents for color grading
        path:       filepath + filenahme for saving
        save:       if True, save plot
        step:       frame steps for animation (f.e. step=5 uses every 5th frame)
        fps:        frames per second for the animation (with step=1)
        title:      Title of the Animation
    """
    # prepare data for animation
    data = []
    person = []
    colors = []

    people_count = int(loader.data['p'].max() - loader.data['p'].min())

    for i in np.arange(frame_start, frame_stop, step):
        data.append(loader.frame(i)[1])
        person.append(loader.frame(i)[0])

    #Set the figure for the animation framework
    fig = plt.figure(figsize = (10,6))
    #creating a subplot 
    ax1 = fig.add_subplot(1,1,1)

    ax1.set_xlim([boundaries[0], boundaries[1]])
    ax1.set_ylim([boundaries[2], boundaries[3]])

    ax1.set_xlabel('x Pos. / cm')
    ax1.set_ylabel('y Pox. / cm ')
    ax1.set_title(title, loc="left")

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
            if (i+1) in ai:
                lobj = ax1.plot([],[], color="black", lw=2)[0]
            else:
                lobj = ax1.plot([],[], color="red", lw=2)[0]
            lines.append(lobj)
            vals.append([[], []])

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    #Using FuncAnimation we need to create an animation function which return and/or done a repetitive action
    def animate(i):
    
        #update data for plotting
        for (per, dat) in zip(person[i], data[i]):
            vals[int(per-1)][0].append(dat[0])
            vals[int(per-1)][1].append(dat[1])
            
        #write new data to line objects
        for lnum, line in enumerate(lines):
            line.set_data(vals[lnum][0], vals[lnum][1])
        return lines

    frames = int(np.floor((frame_stop - frame_start)/step))
    ani = animation.FuncAnimation(fig = fig, func = animate, frames = frames, interval = int(step*1000/fps), blit=True) 
    plt.close(fig)
    
    if save:
        ani.save(path, fps=fps, extra_args=['-vcodec', 'libx264'])
    return ani


#fast data preview
def animatePreview(loader, boundaries, step):
    """ quickly preview Animation in a jupyter notebook, using plotly.

        loader:     loader objekct that stores the data
        boundaries: figure boundaries [xmin, xmax, ymin, ymax]
        step:       frame-step from one image to the next (step=1 means all frames are plotted, step=5 everey 5th)
        """
    import plotly.express as px
    fig = px.scatter(loader.data[(loader.data['f'] % 10) == 0], 
           x="y", y="x", 
           animation_frame="f", animation_group='p', hover_name="p",
           range_x=[boundaries[0], boundaries[1]], range_y=[boundaries[2], boundaries[3]],
           template="plotly_white", title="Animation Preview")
    fig.show()