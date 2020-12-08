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
