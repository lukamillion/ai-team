# Parameter Scan for the Corridor dataset

## Dataset: ug-180-095

### Description
The general parameters used for the scan were

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
model_2 | True | True | False | 8 | EPFL
model_3 | True | False | True | 8 | EPFL
model_4 | False | False | False | 8 | EPFL
model_5 | True | True | True | 8 | [50,20,50]
model_6 | True | True | True | 1 | EPFL

### File Layout

The files param_scan_***_i.npy, i = 0,...,5,p contain the measured speeds (vel) and density (den) in the ROI.
param_scan.npy contains the mean speed / density for the datasets in one array in the shape:

axis 0: different model sets
axis 1: [density_pers, density_pers_err, density_agent, density_agent_err, 
	velocity_pers, velocity_pers_err, velocity_agent, velocity_agent_err]