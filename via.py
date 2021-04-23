# from via import via; via(x)
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import PIL

def via(arr:str or np.ndarray, save_txt:bool = False, size:tuple = (20,20), 
                out:str = 'array_out.txt', normalize:bool = False, 
                color_img :bool = False):
    if isinstance(arr, str):
        im = PIL.Image.open(arr)
        arr = np.asarray(im)
    if isinstance(arr, np.ndarray):
        # (#Images, #Chennels, #Row, #Column)
        if dim == 4:
            arr = arr.transpose(3,2,0,1)
        if dim == 3:
            arr = arr.transpose(2,0,1)
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    dim = arr.ndim 
    fig = plt.figure(figsize=size)

    if save_txt:
        if not out.endswith('.txt'): out += '.txt'
        with open(out, 'w') as outfile:    
            outfile.write('# Array shape: {0}\n'.format(arr.shape))
            
            if dim == 1 or dim == 2:
                np.savetxt(outfile, arr, fmt='%-7.3f')

            elif dim == 3:
                for i, arr2d in enumerate(arr):
                    outfile.write('# {0}-th channel\n'.format(i))
                    np.savetxt(outfile, arr2d, fmt='%-7.3f')

            elif dim == 4:
                for j, arr3d in enumerate(arr):
                    outfile.write('\n\n# {0}-th Image\n'.format(j))
                    for i, arr2d in enumerate(arr3d):
                        outfile.write('# {0}-th channel\n'.format(i))
                        np.savetxt(outfile, arr2d, fmt='%-7.3f')
            else:
                print("Out of dimension!")

    
    if out.endswith('.txt'): out = out.replace('txt', 'png')  
    else : out += '.png'

    if normalize: arr -= np.min(arr); arr /= max(np.max(arr),10e-7);
    if dim == 1 or dim == 2:
        if dim==1: arr = arr.reshape((1,-1))
        fig.suptitle('Array shape: {0}\n'.format(arr.shape), fontsize=30)
        plt.imshow(arr, cmap='jet')
        plt.colorbar()
        fig.savefig(out)

    elif dim == 3:
        if color_img == False:
            x_n = int(np.ceil(np.sqrt(arr.shape[0])))
            fig.suptitle('Array shape: {0}\n'.format(arr.shape), fontsize=30)
            for i, arr2d in enumerate(arr):
                ax = fig.add_subplot(x_n,x_n,i+1)
                im = ax.imshow(arr2d, cmap='jet')
                plt.colorbar(im)
                ax.set_title('{0}-channel'.format(i))
            fig.savefig(out)
        else:
            arr = arr.transpose(1,2,0)
            arr = (arr - np.min(arr))/np.ptp(arr)
            plt.imshow(arr)
            fig.savefig(out)

    elif dim == 4:
        img_n = arr.shape[0]
        x_n = int(np.ceil(np.sqrt(arr.shape[1])))
        outer = gridspec.GridSpec(img_n, 1)
        fig.suptitle('Array shape: {0}\n'.format(arr.shape), fontsize=30)
        for j, arr3d in enumerate(arr):
            inner = gridspec.GridSpecFromSubplotSpec(x_n, x_n, subplot_spec=outer[j],wspace=0.1,hspace=0.3)
            for i, arr2d in enumerate(arr3d):
                ax = plt.subplot(inner[i])
                im = ax.imshow(arr2d, cmap='jet')
                plt.colorbar(im)
                ax.set_title('{0}-Image {1}-channel'.format(j,i))
        
        fig.suptitle('Array shape: {0}\n'.format(arr.shape), fontsize=30)
        if out.endswith('.txt'): out = out.replace('txt', 'png')  
        else : out += '.png'
        fig.savefig(out)

    else:
        print("Out of dimension!")

    
"""
arr = torch.randn(3,28,35)
via(arr, color_img=True)
"""
