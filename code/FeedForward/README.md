# Code - FeedForward

Here you find all codes for the Feed Forward approach. The *.py files provide the helper functions that we use for different experiments. Use the jupyter notebook to run a complete training, validation and testing of a model. 

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
* Helper to measure meanspead and density



