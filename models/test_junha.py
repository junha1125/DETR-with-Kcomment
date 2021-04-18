
# %%
import torch 
checkpoint = torch.load('/workspace/detr/detr/DETR-with-Kcomment/result/eval.pth')

""" 
chechpoint.keys()
dict_keys(['params', 'counts', 'date', 'precision', 'recall', 'scores'])
"""

# %%
