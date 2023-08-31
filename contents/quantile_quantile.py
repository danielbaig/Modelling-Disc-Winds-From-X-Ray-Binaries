import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
pathtohere = Path.cwd()


class QQAnalysis:
    """
    Class storing functions involved in the linearised quantile-quantile plot. This works by first forming the 
    quantile-quantile plot as usual by finding the quantile of the two sets of data and comparing them. It then
    rotates the axis by tau/8 rad and renormalises the distance between the points and the central axis to between 0
    and 0.5 (for plotting convienience).

    compare
        _calculate
        _displayQQ
        _plot
    """

    def __init__(self, data0:np.ndarray,data1:np.ndarray,variables:tuple,name0:str,name1:str,logged_var:set={}):
        """
        Initalise the class.

        Inputs:
            - data0:np.ndarray: Data for the first point.
            - data1:np.ndarray: Data for the second point.
            - variables:tuple: Variables that are included
            - name0:str: Name of the first data point.
            - name1:str: Name of the second data point.
            - logged_var:set: The variables that are logged.
        """

        self.data0 = data0
        self.data1 = data1
        self.variables = variables
        self.name0 = name0
        self.name1 = name1
        self.logged_var = logged_var

        self.numVariables = len(self.variables)

        assert self.data0.shape[0]==self.data1.shape[0]==self.numVariables, f'Sizes of data or variables is wrong or needs to be transposed: data0-{self.data0.shape}, data1-{self.data1.shape}, variables-{self.numVariables}.'



    def _calculate(self):
        """
        Calculates the quartiles for each variable and normalises them.
        """


        epsilon = 1e-12

        # Set entries of zero to a small float if they are going to be logged.
        for var in self.logged_var:
            if var not in self.variables:
                continue
            var = self.variables.index(var)
            self.data0[var][np.where(self.data0[var]<epsilon)] = epsilon
            self.data0[var] = np.log10(self.data0[var])
            self.data1[var][np.where(self.data1[var]<epsilon)] = epsilon
            self.data1[var] = np.log10(self.data1[var])



        # Determine quantiles.
        self.quantileSamples = np.arange(0.01, 1., 1e-2)

        quantileMethod = 'median_unbiased'
        quantiles0 = np.quantile(self.data0,self.quantileSamples,axis=1, method=quantileMethod)
        quantiles1 = np.quantile(self.data1,self.quantileSamples,axis=1, method=quantileMethod)

        # Rotated values.
        self.differences = (quantiles1 - quantiles0)/np.sqrt(2)
        heights = (quantiles0 + quantiles1)/np.sqrt(2)

        # Normalise.
        min_height = np.min(heights, axis=0)[None,:]
        max_height = np.max(heights, axis=0)[None,:]
        self.heights = (heights - min_height) / (max_height - min_height)

        self.differences /= np.max(np.abs(self.differences), axis=0)[None,:]*2



    def _displayQQ(self, exampleDir:str=None):
        """
        Display all qq comparisons on the sample plot.

        Inputs:
            - exampleDir:str: Directory of the example (if not running on two points).
        """


        colours=('r','g','b','m','y','c','pink','orange','purple','brown','silver','lime','gold')

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()

        ax.axhline(0,c='k')
        # Plot points.
        for i in range(self.numVariables):
            ax.scatter(self.heights[:,i], self.differences[:,i], marker='.',
                    label=self.variables[i], c=colours[i],zorder=1)


        # Create appropiate labels
        ax.set_xlabel('proportion along diagonal')
        ax.set_ylabel('difference to diagonal')
        ax.legend(loc='best')

        
        if exampleDir==None:
            restofpath = 'plots/qqcomp.png'
        else:
            restofpath = 'examples/' + exampleDir +  '/qqcomp.png'


        plt.savefig(pathtohere / restofpath, bbox_inches='tight')




    def _plot(self, exampleDir:str=None):
        """
        Display the qq comparison plot.

        Inputs:
            - exampleDir:str: Directory of the example (if not running on two points).
        """

        # Create figure
        fig = plt.figure(figsize=(12,6), dpi=150)
        ax = fig.add_subplot()

        # Plot points.
        for i,var in enumerate(self.variables):
            scat = ax.scatter(self.heights[:,i],i+self.differences[:,i],c=self.quantileSamples,
                                marker='.',zorder=2, s=10.)
            ax.plot(self.heights[:,i],i+self.differences[:,i],zorder=1,c='c')
            ax.axhline(i, c='grey', lw=0.1)

        cbar = plt.colorbar(scat, ax=ax)
        
        # Create appropiate labels
        cbar.ax.set_ylabel('quantile')
        ax.set_yticks(range(self.numVariables), self.variables)
        ax.set_xlabel('proportion along diagonal')
        ax.set_ylabel('wind property')
        ax.set_title(f'Comparing quantiles of point {self.name0} to {self.name1}.')

        
        if exampleDir==None:
            restofpath = 'plots/pointComparison.png'
        else:
            restofpath = 'examples/' + exampleDir +  '/distributionComparison.png'

        plt.savefig(pathtohere / restofpath, bbox_inches='tight')



    def compare(self,exampleDir:str=None):
        """
        Compare the quartiles of two distributions for a number of variables.

        Inputs:
            - exampleDir:str: Directory of the example (if not running on two points).
        """

        self._calculate()
        self._displayQQ(exampleDir)
        self._plot(exampleDir)


