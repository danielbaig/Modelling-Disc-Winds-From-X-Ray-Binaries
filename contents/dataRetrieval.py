import numpy as np

import astropy.io.ascii as io
import re

from pathlib import Path
pathtohere = Path.cwd()


def prepareData(directoryName, variables, isAll:bool=False, isNormalised:bool=False,
                isAltLoc:bool=False, logged_var:set={}):
    """
    Prepare the data for compression.

    Inputs:
        - directoryName:str: Name of the directory.
        - variables:tuple: Variables to include.
        - isAll:bool: Include of the data.
        - isNormalised:bool: Whether to normalise the data.
        - isAltLoc:bool: Is the alternative location (this is temporary).
        - logged_var:set: The variables to log (before normalising, again this is temporary).
    """

    _extra = '../../' if isAltLoc else ''

    # Get data
    dataFilePath = pathtohere / (_extra + 'data/' + directoryName +'/xrb_1820.master.txt')
    data_ast = io.read(dataFilePath)
    data = np.loadtxt(dataFilePath, skiprows=1, dtype=float)

    # Get He1
    dataFilePath = pathtohere / (_extra + 'data/' + directoryName + '/xrb_1820.He.frac.txt')
    _temp_ast = io.read(dataFilePath)
    data_ast['he1'] = _temp_ast['i01']
    _temp = np.loadtxt(dataFilePath, skiprows=1, dtype=float)
    data = np.c_[data,_temp[:,5]]



    if not isAll:
    
        
        # Ensures the data is in the wind.
        converged = np.where((data_ast['converge']==0) & (data_ast['inwind']==0))
        data = data[converged]
        data_ast = data_ast[converged]

        # Gets relavant variables.
        keys = data_ast.keys()
        variableColumns = [keys.index(ele) for i,ele in enumerate(variables) if ele in keys]
        #data_ast = data_ast[includedVariables]
        data = data[:,variableColumns]
        print(f'Included variables:\n{variables}')

        #assert data.shape[1] == np.shape(data_ast.keys())[0], f'Shape of data and data_ast are not the same: {data.shape}!={np.shape(data_ast)}.'


        # Remove problematic entries
        include = np.fromiter(((np.all((d!=np.inf) & (d!=np.nan))) for d in data), dtype=bool)
        data = data[include]
        data_ast = data_ast[include]

        epsilon = 1e-10

        # Set entries of zero to a small float if they are going to be logged.
        for i,var in enumerate(variables):
            if var not in logged_var:
                continue
            data[:,i][np.where(data[:,i]<epsilon)] = epsilon
            data[:,i] = np.log10(data[:,i])

        if isNormalised:

            # Normalise data.
            maxes = np.max(data, axis=0)[None,:]
            mins = np.min(data,axis=0)[None,:]
            data = (data - mins) / (maxes - mins)


            assert np.max(data) == 1., f'Data incorrectly normalised, np.max(data)={np.max(data)} not 1.'

    return data, data_ast





def write_pf_data(line:np.ndarray):
    """
    Write parameter file data to the features.txt file.

    Inputs:
        - line:np.ndarray: Line being altered.
    """
    index = line[0]

    maxFileSize = int(1e+2)
    i = 0
    found = np.asarray([False, False, False, False])
    parameterValues = np.empty_like(found, dtype=float)

    # Get the point's coordinates from the .pf file.
    with open(pathtohere / ('data/' + f'XRB{index}/xrb_1820.pf'),'r') as f:
        while i<maxFileSize:
            line = f.readline()
            if line[0]=='\n':
                i += 1
                continue
            line = re.split('\n|\t| ', line)
            line = list(filter(None, line))

            j = -1
            
            if line[0]=='Wind.mdot(msol/yr)':
                j = 0
            elif line[0]=='Wind.filling_factor(1=smooth,<1=clumped)':
                j = 1
            elif line[0]=='SV.acceleration_length(cm)':
                j = 2
            elif line[0]=='SV.acceleration_exponent':
                j = 3

            if j!=-1:
                parameterValues[j] = float(line[1])
                found[j] = True

            if found.all():
                break



            i += 1
        else:
            raise Exception(f'Maximum file size ({maxFileSize}) reached.')


    with open(pathtohere / 'data/features.txt','r', encoding='utf-8') as f:
        data = f.readlines()



    ## Assumes that the samples are in order already.
    data[int(index)+1] = (f'{index} {parameterValues[0]} {parameterValues[1]} {parameterValues[2]} {parameterValues[3]}'
            + data[int(index)+1][9:])

    with open(pathtohere / 'data/features.txt', 'w', encoding='utf-8') as f:
        f.writelines(data)


    return re.split(',',data[int(index)+1][:-1])




def get_featureFile():
    """
    Get the contents of the feature file. Ensure the feature.txt file is complete. If not, attempt to complete it.

    Outputs:
        - data:np.ndarray: Data from the feature file.
    """


    data = np.loadtxt('data/features.txt', skiprows=1, dtype=str, delimiter=' ')
    
    for i,line in enumerate(data):
        if len(line)!=6 and i!=0:
            raise Exception(f'Entry does not have 6 columns, it is possible spaces have not been'
                            +f' placed between the _?:\n{line}')

        if '_' in line:
            data[i] = write_pf_data(line)
    return data


def transformData(dataIn:np.ndarray):
    """
    Splits the file information into point data and colours.

    Inputs:
        dataIn:np.ndarray: Data from the .txt file.

    Outputs:
        - dataOut:np.ndarray: Point positions.
        - colours:np.ndarray: RGBA values.

    """

    dataOut = dataIn[:,1:-1].astype(float)

    colours = np.asarray([('r','g')[d] for d in dataIn[:,-1].astype(bool)])


    return dataOut.T, colours


