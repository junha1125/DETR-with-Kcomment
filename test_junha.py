#%% [markdown]

# %%
# # Positional Embeding
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
import numpy as np
import pickle

def find_max_indices(list):
    arr = np.array(list)
    min_indices = []
    for i in range(10):
        index = np.argmax(arr)
        min_indices.append(index)
        arr[index] = 0
    return min_indices

def get_name_list(names, max_indices):
    min_names = []
    for index in max_indices:
        min_names.append(names[index])
    with open("max_names.txt", 'w') as f:
        f.write("\n".join(map(str, min_names)))

    
with open("test_Image_id.txt", "rb") as fp:   # Unpickling
    Image_id = pickle.load(fp)

with open("test_val_error.txt", "rb") as fp:   # Unpickling
    val_error = pickle.load(fp)

with open("test_val_losses.txt", "rb") as fp:   # Unpickling
    val_losses = pickle.load(fp)


#%%
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import PIL

max_loss = ['000000558421.jpg',
'000000520832.jpg',
'000000315219.jpg',
'000000049761.jpg',
'000000231125.jpg',
'000000011615.jpg',
'000000565607.jpg',
'000000200839.jpg',
'000000377882.jpg']

max_error = ['000000002587.jpg',
'000000005503.jpg',
'000000006614.jpg',
'000000011615.jpg',
'000000014888.jpg',
'000000015751.jpg',
'000000018833.jpg',
'000000023751.jpg',
'000000025593.jpg']

root = '/dataset/coco/val2017/'

for i in range(9):
    max_loss[i] = root + max_loss[i]
    max_error[i] = root + max_error[i]

_, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten()
for img, ax in zip(max_error, axs):
    im = PIL.Image.open(img)
    ax.imshow(im)
plt.show()

# %%
