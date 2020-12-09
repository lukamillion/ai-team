# Data Format

### nn_scan.npy

axis 1: nn_values (1, ..., 8, 12)
axis 0: mean / err entries
[density_pers, density_pers_err, density_agent, density_agent_err,
 velocity_pers, velocity_pers_err, velocity_agent, velocity_agent_err]


#### nn_scan_***_i.npy, i = 0,...,8,p

raw speed (vel) / density (den) data
The index i corresponds to the nearest neighbor number as #nn = i+1