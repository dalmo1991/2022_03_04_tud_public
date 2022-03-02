import numpy as np

def nse(obs, sim):
    obs_mean = np.mean(obs)
    nse = 1 - np.sum((obs-sim)**2)/np.sum((obs-obs_mean)**2)
    return nse