#%% [markdown]
# # Positional Embeding

# %%
import numpy as np
d_vector = 256
num_vector = 625
PEs = np.empty((num_vector,d_vector))

period = 1000
for i in range(num_vector):
    if i%2 == 0:
        w = 1/ period**(2*i/d_vector)
        pos = np.arange(256)
        PEs[i] = np.sin(pos*w)
    if i%2 != 0:    
        w = 1/ period**(2*i/d_vector)
        pos = np.arange(256)
        PEs[i] = np.cos(pos*w)

%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(25, 23), dpi=80, facecolor='black')
imgplot = plt.imshow(PEs)
plt.colorbar()

# %%
