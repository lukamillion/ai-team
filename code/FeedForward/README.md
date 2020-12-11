# Code - FeedForward

Here you find all codes for the Feed Forward approach. The *.py files provide the helper functions that we use for different experiments. Use the jupyter notebook (AgentTraining.ipynb) to run a complete training, validation and testing of a model. 

## Library

### data_loader.py
    
* Load person trajectories from the given datasets. 
* Extract Train/Validation/Test split for a feed forward network
* Manipulate people in the dataset
* Save a dataset


### data_viewer.py
* Read a data file stored with the data_loader
* Indicate Persons and Agents
* Indicate experimental setup 
* Visualize in different ways:  
  * Graph a trajectory of all people
  * Animate the trajectory of every person
  * Animate the location of every person 


### hdf5_utils.py
* Save and Load Database splits with parameters
* Save and Load Torch models with parameters
* graph content of a hdf5 file


### simulation.py
* Agent class with using trained model
* Engine class to run simulations 
* Helper to measure meanspeed and density

### visualizatioin.py
* Creating heatmaps and plots for presentation and report.

## Installation
In order to run the code on your computer, the following programs / environments are needed:

1. Install Python 3 (we recommend installing it via [Anaconda](https://www.anaconda.com))
2. Install jupyter notebook (This is part of anaconda or python -m pip install jupyter)
3. Install [Pytorch](https://pytorch.org) (if possible with CUDA support) 
4. Install various additional packages
    * [Progressbar](https://pypi.org/project/progressbar2/)
    * [HDF5](https://docs.h5py.org/en/stable/)
    * [Pandas](https://pandas.pydata.org/)

Code was developed and tested on the following system but should run platform independant as of our knowlage. 

    Ubuntu 20.04.1
    Python 3.8.3

    torch==1.7.0
    pandas==1.0.5
    numpy==1.18.5
    matplotlib==3.2.2
    h5py==2.10.0
    progressbar2==3.37.1

    cudatoolkit==11.0.221




