# Agent-Based Modeling and Social System Simulation 2019

> * Group Name: A(I)-Team
> * Group participants names: Michael Eichenberger, Isabel Heidtmann, Alexander Jürgens, Luka Milanovic, Dominique Zehnder
> * Project Title: AI in Pedestrian Dynamics

## General Introduction

As the title implies, our project exists in the overlap of two interesting, but quite different, sciences - machine learning and pedestrian dynamics. Since machine learning using neural networks is a relatively new and quickly growing field, we thought that it would be nice to have at least some first hand experience with it. And because the context of our project is more about applying neural networks and less about working on the theory behind it, it was a good opportunity for us to gather our first experiences. Pedestrian dynamics on the other hand intrigued us mostly because the system itself, containing not just one, but many human minds, seems infinitely complex. But still it can be described with surprisingly simple models.

While searching for a more specific goal, we have come across a paper published in the Journal of Intelligent Transportation Systems[[1]]. It is about using neural networks to predict the speed of pedestrians. We wanted to take it a step further and also try to predict the velocity (speed and direction) of pedestrians.

This approach is very different to most historical approaches which view pedestrians as something akin to particles (social forces model) or fluids. We believe that our more data driven approach could be used to view this problem from a different angle. A model a bit more refined than ours could be used to test out different geometries for spaces where pedestrians should move in. This information could then be used to optimize those spaces and improve metrics, like maximum flow, dwelling time or to predict pedestrian trajectories in an autonomous vehicle. 




## The Model

As our model for the decision-making pedestrian we chose to use feed forward neural networks. As our approach is agent-based, each pedestrian was represented by his own neural network. The agents were placed in an empty 2d world and in each discrete timestep each agent predicted his position in the next timestep. The 2d space was continous and each agent occupied an infinitesimal area. As inputs theneural networks each used the position and velocity of the represented agent and his n nearest neighbours.

For training the models used recorded data of real people moving in a laboratory setting. The network had to predict the n-th next position of the recorded person, we called this downsampling. This means that our training process was fundamentally different from the later application of the agents.

We achieved the best results by tweaking the inputs and outputs of the network, using different datasets to train it, varying the amount of downsampling and modifying the training hyperparameters.




## Fundamental Questions

The main question of this project was to see if agent based neural networks can be used to accurately describe pedestrian dynamics. To see if our model achieved this feat we had to compare our model to the recorded human data in a pedestrian dynamics context and access it's fitness. For this we generated velocity-flow and density-flow diagrams for both our model moving int he 2d space described above and the recorded data. For this question to be answere satisfyingly it was also important to look at the weaknesses and breakdown points of our model. In what extreme situations would our neural networks not be able to predict human behaviour anymore?


## Expected Results

We expected our model to stand up to a visual comparison to real human data and when analyzed quantitatively to also show the characteristic velocity-flow and density-flow relationships seen in pedestrian behaviour. 


## References

WIRD AM ENDE GEMACHT


## Research Methods

Agent-Based Model and data based machine learning using neural networks


## Other

### Dataset: 
*Data archive of experimental data from studies about pedestrian dynamics* ,
[Forschungszentrum Jülich](https://ped.fz-juelich.de/database/doku.php)