import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from ..dataRetrieval import prepareData

from pathlib import Path
pathtohere = Path.cwd()



def get_sortedData(maxIndex:int,variables:tuple, startIndex:int=0) -> np.ndarray:
    """
    Gets the data and includeds all points that are in all pf data points.

    Inputs:
        - maxIndex:int: Maximum index to include.
        - variables:tuple: Variables to include.
        - startIndex:int: Index to start at.

    Outputs:
        - data:np.ndarray: Formatted and sorted data in the form (pfIndex, pointIndex, variableIndex)
    """

    logged_var = {'xcen','zcen','ne','h1','he2','o6','c4','n5','t_e','t_r','xi','he1'}

    notExist = []

    
    data = [[] for i in range(startIndex,maxIndex+1)]
    data_ij = [[] for i in range(startIndex,maxIndex+1)]


    # Get normalised data.
    for i in range(startIndex,maxIndex+1):
        data[i-startIndex] = prepareData(f'XRB{i}', variables,
                        isNormalised=False,isAltLoc=False, logged_var=logged_var)[0]
        data_ij[i-startIndex] = set(map(tuple,np.asarray(prepareData(f'XRB{i}',('i','j'),
                                        isAltLoc=False)[0], dtype=int)))

        # Get maximum and minimum.
        if i==startIndex:
            maxData = np.max(data[i-startIndex],axis=0)
            minData = np.min(data[i-startIndex],axis=0)
        else:
            trialMaxData = np.max(data[i-startIndex],axis=0)
            trialMinData = np.min(data[i-startIndex],axis=0)

            whereMax = np.where(trialMaxData > maxData)
            whereMin = np.where(trialMinData < minData)
            maxData[whereMax] = trialMaxData[whereMax]
            minData[whereMin] = trialMinData[whereMin]




    # Normalise.
    largestRange = maxData - minData
    for i,d in enumerate(data):
        data[i] = (d - minData) / largestRange


    
    # Find common data points.
    intersection = set.intersection(*[data_ij[i] for i in range(maxIndex+1 - startIndex)])

    include = [[] for k in range(startIndex,maxIndex+1)]
    for k in range(maxIndex+1 - startIndex):
        include[k] = np.asarray([i in intersection for i in sorted(list(data_ij[k]))])
        if k !=0:
            continue



    # Only include common data points.
    for i in range(maxIndex+1 - startIndex):
        
        data[i] = data[i][include[i]]



    return np.asarray(data)


def save_sortedData(data:np.ndarray):
    """
    Save sorted data.

    Inputs:
        - data:np.ndarray: Data to save in form (pfIndex, pointIndex, variableIndex)
    """

    pathtohere = Path.cwd()

    np.savetxt(pathtohere / 'contents/stringMinimiser/data/potentialPositions.txt', data.flatten())

    np.savetxt(pathtohere / 'contents/stringMinimiser/data/potentialPositionsDims.txt', np.asarray(data.shape,int))




def testChanges(data:np.ndarray, variables:tuple, maxIndex:int, testVariable:int):
    """
    Test the change in each point between each parameter file to ensure correct and natural mapping.

    Inputs:
        - data:np.ndarray: Supposedly organised data.
        - variables:tuple: Variables that were used.
        - maxIndex:int: Maximum pf index.
        - testVariable:int: Variable to test.
    """

    fig = plt.figure(figsize=(16,16), dpi=200, tight_layout=True)
    fig.suptitle(f'Variable = {variables[testVariable]}')

    for i in range(maxIndex):
        for j in range(maxIndex):
            if i>j:
                continue

            ax = fig.add_subplot(maxIndex, maxIndex, i+j*maxIndex+1)
            ax.scatter(data[i,:,testVariable], data[j,:,testVariable], s=1e-1)

            # Create appropiate labels.
            if j==maxIndex-1:
                ax.set_xlabel(i)
            if i==0:
                ax.set_ylabel(j)


    plt.savefig(pathtohere / f'contents/stringMinimiser/plots/tests/test_{variables[testVariable]}.png',
    bbox_inches='tight')




def get_densityNormalisation(maxIndex:int, testingIndex:int):
    """
    Calculates the density of points about each pfps point.

    Inputs:
        - maxIndex:int: Maximum point index.
        - testingIndex:int: Point being tested against.

    Outputs:
        - densities:np.ndarray: Densities of points if pfps.
    

    """

    std = 0.1
    logged_indicies = (0,1,2)

    potentialPositions_pf = np.loadtxt(pathtohere / 'data/features.txt', delimiter=' ',
                                    dtype=float, skiprows=1,usecols=(1,2,3,4))
    
    # Log.
    for i in range(4):
        if i not in logged_indicies:
            continue
        potentialPositions_pf[:,i] = np.log10(potentialPositions_pf[:,i])


    # Normalise.
    maxPos = np.max(potentialPositions_pf, axis=0)
    minPos = np.min(potentialPositions_pf, axis=0)

    potentialPositions_pf = (potentialPositions_pf - minPos[None, :]) / (maxPos - minPos)[None, :]



    densities = np.empty(maxIndex)
    for i in range(maxIndex):
        distances = np.linalg.norm(potentialPositions_pf - potentialPositions_pf[i], axis=1)
        densities[i] = np.sum(np.exp(-distances*distances / std / 2))

    densities = np.concatenate((densities[:testingIndex], densities[testingIndex+1:])) / (maxIndex-1)


    return densities


def main():
    cutoff = 10000

    print('\nOrganising data for string minimisation.')

    testingIndex = 15
    maxIndex = 15
    variables = ('ne','t_e','t_r','h1','xi','he1')

    data = get_sortedData(maxIndex,variables)[:,:cutoff]
    data_test = data[testingIndex]
    data = np.concatenate((data[:testingIndex], data[testingIndex+1:]))

    densities = get_densityNormalisation(maxIndex, testingIndex)
    
    # Save data.
    save_sortedData(data)
    np.save(pathtohere / 'contents/stringMinimiser/data/testDataPoints.npy', data_test)
    np.savetxt(pathtohere / 'contents/stringMinimiser/data/densities.txt', densities)


    print('\nCreating testing plots.')
    for i in tqdm(range(len(variables))):
        testChanges(data, variables, maxIndex, i)
    print('Tests available at:', pathtohere / 'contents/stringMinimiser/plots/tests/')






