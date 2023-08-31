import sys

import astropy.io.ascii as io
import argparse


from contents.dataRetrieval import *
from contents.pairwise import *
from contents.primaryAnalysis import *
from contents.quantile_quantile import *
from contents.bhtsne.execute_bhtsne import *

from contents.lineMinimiser import MCAnalysis
from contents.lineMinimiser import organiseData

from examples.qq_flattening import qq_flattening

from pathlib import Path
pathtohere = Path.cwd()



def getArguments(*args):
    """
    Get the arguments from the command line.
    """

    parser = argparse.ArgumentParser(
                    prog='DataAnalysis',
                    description="Analyse the wind data in a directory, or compare different directories' data.",
                    epilog='This code is still being developed so expect bugs.')

    parser.add_argument('directoryName', type=str,
            help="The name of the directory being analysed or `all' if all directories' data are being compared. `line' if you want to perform data analysis on the line minimisation results. Use `prepareLine' to prepare the data for the line minimiser.")
    parser.add_argument('-s','--includeSpectrum', default=True, const=False, action='store_const',
            help='Whether to recompute the spectrum plots.')
    parser.add_argument('-p','--includePairwise', default=True, const=False, action='store_const',
            help='Whether to recompute the pairwise correlation plot.')
    parser.add_argument('-f','--includeFlowField', default=True, const=False, action='store_const',
            help='Whether to recompute the flow field.')
    parser.add_argument('-b','--includeBHTSNE', default=True, action='store',
            help='Whether to include the calulation (True) or just display (display) of the bhtsne.')
    parser.add_argument('-c','--compareTo', default=None, help='Second sample to compare to.')
    parser.add_argument('-e', '--example', default=False, const=True, action='store_const', 
            help='Look at an example.')

    return parser.parse_args()





def compareAll():
    """
    Compare all data points.
    """
    print('\nComparing all samples.')

    data = get_featureFile()
    data, colours = transformData(data)

    pairwise = Pairwise(data, ('mdot','ff','a_length','a_exp'), colours=colours, logged_var = {'mdot','ff'},
            isUpperOn=False, isDiagonalOn=False, isBuffer=True)
    pairwise.createPairwiseFigure()




def singleAnalyse(args, variables:tuple, logged_var:set):
    """
    Analyse a single data point.

    Inputs:
        - args: Arguments given by the user.
        - variables:tuple: Variables to analyse.
        - logged_var:set: Logged variables.
    """


    dataVisualiser = DataVisualiser(args.directoryName, variables)

    if args.includeSpectrum:
        print('\nCreating spectrum.')

        spectrumData = io.read('{}.spec'.format(pathtohere / ('data/' + args.directoryName + '/xrb_1820')))
        print(spectrumData.keys())
        
        dataVisualiser.plot_spectra(spectrumData)
        dataVisualiser.plot_zoomedSpectrum(spectrumData, wmin=5850,wmax=5900,smooth=21)



    if args.includePairwise:
        print('\nCreating pairwise plot.')

        data, data_ast = prepareData(args.directoryName, variables)

        pairwise = Pairwise(data.T, variables, args.directoryName, colours='t_e',
                logged_var=logged_var, num_std=None)
        pairwise.createPairwiseFigure()

    if args.includeFlowField:
        print('\nCreating flowfield.')

        all_data_ast = io.read(pathtohere / ('data/' + args.directoryName + '/xrb_1820.master.txt'))
        dataVisualiser.flowField(all_data_ast)

    if args.includeBHTSNE!='False':
        print('\nCreating bhtsne.')
        isCalcBHTSNE = True if args.includeBHTSNE else False
        compressedData = calculate_bhtsne(args.directoryName, variables, isCalculating=isCalcBHTSNE,cmap='t_e')


def compareTwo(args, variables:tuple, logged_var:set={}):
    """
    Compare two data points.

    Inputs:
        - args: Arguments given by the user.
        - variables:tuple: Variables to analyse.
        - logged_var:set: Logged variables.
    """

    print('\nGenerating quantile-quantile plot.')

    data0, data_ast0 = prepareData(args.directoryName, variables)
    data1, data_ast1 = prepareData(args.compareTo, variables)

    qq = QQAnalysis(data0.T, data1.T,variables,name0=args.directoryName,name1=args.compareTo,
            logged_var=logged_var)


    qq.compare()
    
    # Plot specific comparisons.
    __, all_data_ast0 = prepareData(args.directoryName, variables, isAll=True)
    __, all_data_ast1 = prepareData(args.compareTo, variables, isAll=True)

    pointComparer = PointComparer(all_data_ast0, all_data_ast1,name0=args.directoryName,name1=args.compareTo,
            logged_var=logged_var)
    print('\nCreating zoomed plot.')
    pointComparer.plot_pairs(variables)
    print('\nCreating mini-big composite plot.')    
    pointComparer.compositeComparePair()


   
def main(*args, **kwds):
    """
    Main function.
    """
    print('START')

    args = getArguments(args)

    variables = ('v_x', 'v_y', 'v_z', 'ne', 't_e', 't_r', 'h1', 'xi', 'he1')
    logged_var = {'xcen','zcen','ne','h1','he2','o6','c4','n5','t_e','t_r','xi', 'he1'}



    if not args.example:
        if args.directoryName=='all':
            compareAll()

        elif args.directoryName=='line':
            MCAnalysis.main()
        elif args.directoryName=='prepareLine':
            organiseData.main()

        elif args.compareTo==None:
            singleAnalyse(args,variables,logged_var)

        else:
            compareTwo(args, variables, logged_var)

    elif (directoryName:=args.directoryName.split('.')[0])=='qq_flattening':
        example_dists, data0, data1 = qq_flattening.generateData()


        qq = QQAnalysis(data0,data1,example_dists, name0='normal', name1='other')
        qq.compare(directoryName)


    else:
        raise Exception('Invalid example name.')




    return 0






if __name__=='__main__':
    main()


