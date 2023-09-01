import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as io

from tqdm import tqdm

from . import bhtsne
from ..dataRetrieval import prepareData

from pathlib import Path
pathtohere = Path.cwd()



def plotTSNE(directoryName:str,data_ast,compressedData:list, perplexities:tuple, label:str=None):
    """
    Plots the t-SNE for different perplexities.

    Inputs:
        - data_ast: Astropy arranged data.
        - compressedData:np.ndarray: Data compressed to 2D with differnt perplexities.
        - perplexities:tuple: Perplexities used.
        - label:str: Data to use for the colourbar.
    """


    nrows = compressedData.shape[1]

    # Colour by most prominant ion.
    if label==None:
        maxIndex = np.argmax((data_ast['h1'],data_ast['he2'],
                               data_ast['c4'],data_ast['n5'],
                               data_ast['o6']), axis=0)


        mostProminantIon = np.empty(nrows,dtype=int)
        for i in range(nrows):
            mostProminantIon[i] = maxIndex[i]

        labels = mostProminantIon

    else:    
        labels = data_ast[label]

    fig = plt.figure(figsize=(12,8), dpi=200, tight_layout=True)
    for i,perplexity in enumerate(perplexities):
        # https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
        ax = fig.add_subplot(-len(perplexities)//-2,2,i+1)
        im = ax.scatter(compressedData[i][:,0], compressedData[i][:,1],20,labels,cmap="coolwarm")
        cbar = fig.colorbar(im)

        
        # Creates appropiate labels
        ax.set_title(f'perplexity={perplexity}')
        ax.set_xlabel(r'$x_i$')
        ax.set_ylabel(r'$x_j$')
        cbar.ax.set_ylabel(label)

    plt.savefig(pathtohere / ('data/' + directoryName +  '/dataAnalysis/tsne.png'), bbox_inches='tight')



def calculate_bhtsne(directoryName:str, variables:tuple, isCalculating:bool=True, cmap:str=None):
    """
    Calculate and display the bhtsne for different perplexcity values.

    Inputs:
        - directoryName:str: name fo the point to analyse.
        - variables:tuple: Variables to analyse.
        - isCalculating:bool: Whether to recalculate the tsne or just display it.
        - cmap:str: Colourmap variable.
    """


    data, data_ast = prepareData(directoryName, variables)
    # Normalise data.
    maxes = np.max(data, axis=0)[None,:]
    mins = np.min(data,axis=0)[None,:]
    data = (data - mins) / (maxes-mins)

    compressedData = []
    # Define perplexities to try.
    perplexities = np.linspace(15,min((data.shape[0]-1)//3,100),4,dtype=int)

    if isCalculating:
        # tsne calculation
        for perplexity in tqdm(perplexities):
            compressedData.append(bhtsne.run_bh_tsne(data, initial_dims=data.shape[1],
                                                    max_iter=int(1e+3), perplexity=perplexity, verbose=False))
        compressedData = np.asarray(compressedData)
        # Save data.
        with open(pathtohere / ('data/' + directoryName + '/dataAnalysis/compressedData.npy'), 'wb') as f:
            np.save(f, compressedData)
    else:
        with open(pathtohere / ('data/' + directoryName + '/dataAnalysis/compressedData.npy'), 'rb') as f:
            compressedData = np.load(f)

    
    plotTSNE(directoryName,data_ast,compressedData, perplexities, cmap)

