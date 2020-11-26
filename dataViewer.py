import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import rc
rc('animation', html='html5')

"""
following additional software is needed for this code to work:

    - for animatePreview: plotly
"""

#plot trajectories
def plotTraj(loader, boundaries, people=None, ai=None, legend=False, wall=False, cor=False, title="Trajectories", path="trajectories.png", save=False):
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
        people = loader.data['p'].unique()


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

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('x Pos. / cm')
    ax1.set_ylabel('y Pox. / cm ')
    ax1.set_title(title, loc="left")

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
        people, temp = loader.frame(i, ret_vel=False, with_id=False)
        data.append(temp)
        ai_data.append(temp[np.isin(people, ai)])

        
    #Set the figure for the animation framework
    fig = plt.figure(figsize = (10,6))
    #creating a subplot 
    ax1 = fig.add_subplot(1,1,1)

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

    #Using FuncAnimation we need to create an animation function which return and/or done a repetitive action
    def animate(i):
        scat.set_offsets(data[i])
        scat_ai.set_offsets(ai_data[i])
        return scat,

    frames = int(np.floor((frame_stop - frame_start)/step))
    
    ani = animation.FuncAnimation(fig = fig, func = animate, frames =frames, interval = int(step*1000/fps), blit=True)
    plt.close(fig)
    
    if save:
        if useFFMPEG:
            writer = animation.FFMpegWriter(fps=fps/step, extra_args=['-vcodec', 'libx264'])
        else:
            writer = animation.PillowWriter(fps=fps/step, extra_args=['-vcodec', 'libx264'])
        ani.save(path, writer=writer)
    return ani



# Trajectory animation
def animateTraj(loader, frame_start, frame_stop, boundaries, ai=None, path="traj_anim.gif", save=False, step=1, fps=16, title="Trajectory Animation", useFFMPEG=False):
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
        useFFMPEG:  use FFMPEG writer instead of Pillow
    """
    # prepare data for animation
    data = []
    person = []
    colors = []

    p_ = loader.data['p'].unique()
    people_count = int(p_[p_ < 1000].max())
    print(people_count)

    for i in np.arange(frame_start, frame_stop, step):
        data.append(loader.frame(i, ret_vel=False, with_id=False)[1])
        person.append(loader.frame(i, ret_vel=False, with_id=False)[0])

    #Set the figure for the animation framework
    fig = plt.figure(figsize = (10,6))
    #creating a subplot 
    ax1 = fig.add_subplot(1,1,1)

    ax1.set_xlim([boundaries[0], boundaries[1]])
    ax1.set_ylim([boundaries[2], boundaries[3]])


    ax1.set_aspect('equal', adjustable='box')
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

    #Using FuncAnimation we need to create an animation function which return and/or done a repetitive action
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

        loader:     loader objekct that stores the data
        boundaries: figure boundaries [xmin, xmax, ymin, ymax]
        step:       frame-step from one image to the next (step=1 means all frames are plotted, step=5 everey 5th)
        """
    import plotly.express as px
    fig = px.scatter(loader.data[(loader.data['f'] % 10) == 0], 
           x="x", y="y", 
           animation_frame="f", animation_group='p', hover_name="p",
           range_x=[boundaries[0], boundaries[1]], range_y=[boundaries[2], boundaries[3]],
           template="plotly_white", title="Animation Preview")
    fig.show()