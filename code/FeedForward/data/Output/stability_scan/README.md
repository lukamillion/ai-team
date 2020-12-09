
# Scan Information

In this scan, a model was trained with the same parameters 10 times. The goal is to see how stable the macroscopic dynamic parameters (speed, density) are in respect to model training, i.e. what is the random influence of training on the model quality.

## Parameters

	- nearest neighbors: 6
  	- shuffle: True
  	- Mode: "wrap"
  	- step_nr: 1
  	- epochs: 15
  	- batch_size: 10
  	- learning rate: 1e-3
  	- decay: 0.1
  	- decay_step: 5

Scan parameters:

model | velocity as Input | with neighbor velocity | truth with velocity | downsampe step | structure
------| ------------------|------------------------|---------------------|----------------|-----------
model_1 | True | True | True | 8 | EPFL


## File Layout

The files stab_scan_***_i.npy, i = 0,...,5,p contain the measured speeds (vel) and density (den) in the ROI.
stab_scan.npy contains the mean speed / density for the datasets in one array in the shape:

axis 0: different model sets
axis 1: [density_pers, density_pers_err, density_agent, density_agent_err, 
	velocity_pers, velocity_pers_err, velocity_agent, velocity_agent_err]