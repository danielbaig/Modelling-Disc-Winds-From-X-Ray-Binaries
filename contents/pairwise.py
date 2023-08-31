import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde, spearmanr

from pathlib import Path
pathtohere = Path.cwd()


class Pairwise:
    """
    Class for functions associated with generating a pairwise correlation plot. The bottom corner consists of
    scatter plot, the diagonal histograms and the upper corner are density plots with contours.

    createPairwiseFigure
        densityPlot
        _labelling
        _axesFormatting
            get_axesLimits


    """

    labelSize = 16
    tickSize = 12


    def __init__(self, data:np.ndarray, variables:tuple, directoryName:str=None, colours:str='b',
            logged_var:set={}, directoryIsPath:bool=False, isBuffer:bool=False,
            isLowerOn:bool=True, isDiagonalOn:bool=True, isUpperOn:bool=True, num_std:float=None):
        """
        Initalises the class.

        Inputs:
            - data:np.ndarray: 2D array of each data entry with information for each variable component. Each row
                               is a new data point.
            - variables:tuple: The name of each variable.
            - directoryName: The directory to store the image. If None, just show the image.
            - colours:str: The colours of the points in the scatter plot or the variable to use as a colour map.
            - logged_var:set: The variables that are to be logged.
            - directoryIsPath:bool: Whether the directory variable is in fact a path.
            - isBuffer:bool: Whether to have an additional buffer around the data.
            - isLowerOn:bool: Whether to show the lower corner scatter plot.
            - isDiagonalOn:bool: Whether to show the diagonal histograms.
            - isUpperOn:bool: Whether to show the upper density plots.
            - num_std:float: Number of standard deviations from the mean to include. If None, include all data.
        """

        assert data.shape[0]==len(variables), f'Number of columns of data does not equal number of variables: {data.shape[0]}!={len(variables)}'
        if num_std!=None:
            assert num_std>0, f'Number of standard deviations must be greater than zero but is {num_std}.'

        self.data = data
        self.variables = variables
        self.directoryName = directoryName
        self.colours = colours
        self.logged_var = logged_var
        self.directoryIsPath = directoryIsPath
        self.isBuffer = isBuffer
        self.isLowerOn = isLowerOn
        self.isDiagonalOn = isDiagonalOn
        self.isUpperOn = isUpperOn
        self.num_std = num_std

        self.numVariables = len(self.variables)
        self.pointSize = 1e+4/(np.sqrt(self.data.shape[1])*self.numVariables*self.numVariables)
 
        # Define colourmap of points dependency.
        if isinstance(self.colours,str) and self.colours in variables:
            cmap_index = self.variables.index(self.colours)
            self.colours = ((self.data[cmap_index] - np.min(self.data[cmap_index])) 
                            / (np.max(self.data[cmap_index]) - np.min(self.data[cmap_index])))



        epsilon = 1e-10
        # Set entries of zero to a small float if they are going to be logged.
        for i,var in enumerate(self.variables):
            if var not in self.logged_var:
                continue
            self.data[i][np.where(self.data[i]<epsilon)] = epsilon
            self.data[i] = np.log10(self.data[i])


    def createPairwiseFigure(self):
        """
        Displays a plot of the correlation between all of the variables.

        Uses https://python-graph-gallery.com/86-avoid-overlapping-in-scatterplot-with-2d-density/

        Inputs:
            - is_std_range:bool: Whether to resrict the data to only the central standard deviations on each axes.
        """

                
        # Create the figure.
        fig = plt.figure(figsize=(16,16),dpi=300,tight_layout=True)
        if self.num_std!=None:
            fig.suptitle(r'$\mu\pm$' + str(self.num_std) + r'$\sigma$ excluding zeros in the data',
                    fontsize=self.labelSize)



        for i,var1 in enumerate(self.variables):
            for j,var2 in enumerate(self.variables):
                if (i<j and not self.isLowerOn) or (i==j and not self.isDiagonalOn) or (i>j and not self.isUpperOn):
                    continue

                ax = fig.add_subplot(self.numVariables, self.numVariables, i+j*self.numVariables+1)
                
                x = self.data[i]
                y = self.data[j]

                ax = self._labelling(ax, x,y, i,j)
                ax = self._populateSubplot(ax,x,y,i,j)

        
        if self.directoryIsPath:
            saveRoot = self.directoryName
        elif self.directoryName == None:
            saveRoot = 'plots/pfPairwise.png'
        else:
            saveRoot = 'data/' + self.directoryName +'/dataAnalysis/pairwise.png'

            try:
                os.mkdir(pathtohere / ('data/' + self.directoryName + '/dataAnalysis'))
            except:
                print("Directory 'dataAnalysis' already exists.")


        
        

        plt.savefig(pathtohere / saveRoot, bbox_inches='tight')


    def _populateSubplot(self,ax,x:np.ndarray,y:np.ndarray,i:int,j:int):
        """
        Populate the subplot with the relevant type of plot.

        Inputs:
            - ax: Instance of the axes.
            - x:np.ndarray: x-data.
            - y:np.ndarray: y-data.
            - i:int: Horizontal index.
            - j:int: Vertical index.

        Outputs:
            - ax: Modified axis.
        """

        (ax, x_min, x_max, y_min, y_max) = self._axesFormating(ax, x, y, i, j)

        # Histograms
        if i==j and self.isDiagonalOn:
            # Displays histogram           
            counts, bins = np.histogram(x, bins=10)
            ax.hist(bins[:-1], bins, weights=counts, color='purple')


        elif i<j and self.isLowerOn:
            # Scatter plot
            ax.scatter(x,y,marker='.',s=self.pointSize, alpha=0.9,c=self.colours)
        elif self.isUpperOn:
            # Density plot
            ax = self.densityPlot(ax,x,y, x_min, x_max, y_min, y_max)

        return ax



    @staticmethod
    def densityPlot(ax, x:np.ndarray, y:np.ndarray,
                                  x_min:float, x_max:float, y_min:float, y_max:float):
        """
        Create a density plot for the pairwise correlation figure.

        Inputs:
            - ax: Instance of the axes.
            - x:np.ndarray: Horizontal components.
            - y:np.ndarray: Vertical components.
            - x_min:float: Minimum value in the horizontal component.
            - x_max:float: Maximum value in the horizontal compoment.
            - y_min:float: Minimum value in the vertical component.
            - y_max:float: Maximum value in the vertical component.

        Outputs:
            - ax: Altered instance of the axes.
        """

        nbins = 20
     

        K = gaussian_kde([x, y])
        xi, yi = np.mgrid[x_min:x_max:nbins*1j,
                          y_min:y_max:nbins*1j]
        zi = K(np.vstack([xi.flatten(), yi.flatten()]))
        axesRanges = (x_min, x_max, y_min, y_max)
        ratio = 1.*(axesRanges[1] - axesRanges[0]) / (axesRanges[3] - axesRanges[2])

        assert ratio>0, 'Epsilon is not low enough.'
        

        # Creates contour plot
        im = ax.imshow(zi.reshape(xi.shape).T, cmap='Oranges',
                       origin='lower', extent=axesRanges,
                       aspect=ratio, interpolation='bicubic')
        ax.contour(xi, yi, zi.reshape(xi.shape), 4, colors='cyan',
                   alpha=0.5, origin='lower' )

        return ax

    def _labelling(self,ax, x:np.ndarray,y:np.ndarray, i:int,j:int):
        """
        Creates appropiate labels.

        Inputs:
            - ax: Instance of the plot.
            - x:np.ndarray: Horizontal components.
            - y:np.ndarray: Vertical components.
            - i:int: Horizontal index.
            - j:int: Vertical index.
            - logged_var:set: The set of logged variables.

        Outputs:
            - ax: Altered instance of the axes.
        """

        # Set y labels
        if i==0:
            label = f'log({self.variables[j]})' if self.variables[j] in self.logged_var else self.variables[j]
            labelSizeAdj = 4 if len(label) > 7 else 0
            ax.set_ylabel(label, color='purple',
                          fontsize=self.labelSize-labelSizeAdj,
                          fontweight='normal')

        # Set x labels (above plot)
        if ((j==0 and self.isUpperOn) 
                or (i==j and not self.isUpperOn and self.isDiagonalOn)
                or (j-i==1 and not self.isUpperOn and not self.isDiagonalOn)):
            label = f'log({self.variables[i]})' if self.variables[i] in self.logged_var else self.variables[i]
            labelSizeAdj = 4 if len(label) > 7 else 0
            ax.set_title(label, color='purple',
                         fontsize=self.labelSize-labelSizeAdj)

        # Put axes labels in scientific notation.
        try:
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        except:
            pass

        return ax


    def _axesFormating(self,ax, x:np.ndarray, y:np.ndarray, i:int, j:int):
        """
        Formats the axes of the pairwise correlation plot.

        Inputs:
            - ax: Instance of the plot.
            - x:np.ndarray: Horizontal components.
            - y:np.ndarray: Vertical components.
            - i:int: Horizontal index.
            - j:int: Vertical index.


        Outputs:
            - ax: Altered instance of the axes.
        """



        # Tick size [https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller]
        ax.xaxis.set_tick_params(labelsize=self.tickSize)
        ax.yaxis.set_tick_params(labelsize=self.tickSize)



        x_min,x_max,y_min,y_max = self.get_axesLimits(x,y)

        ax.set_xlim(x_min, x_max)
        if i!=j:
            ax.set_ylim(y_min, y_max)
      
            
        # Remove tick labels on non-edge figures.
        if i!=0:
            ax.set_yticks([])
        if j!=self.numVariables-1:
            ax.set_xticks([])

        
        ax.set_box_aspect(1)

        if j==0:
            print(f'Range of: {self.variables[i]} ({x_min},{x_max})')

        # Colour axes
        try:
            correlation_coeff = spearmanr(x,y)
        except:
            correlation_coeff = (1,0)
        colour = 'g' if abs(correlation_coeff[0]) > 0.8 else 'r'
        # https://stackoverflow.com/questions/7778954/elegantly-changing-the-color-of-a-plot-frame-in-matplotlib
        if i!=j:
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)


        return ax, x_min, x_max, y_min, y_max


    def get_axesLimits(self, x:np.ndarray,y:np.ndarray):
        """
        Gets the range of the axis for a subplot.

        Inputs:
            - x:np.ndarray: x-data.
            - y:np.ndarray: y-data.

        Outputs:
            - x_min:float: Minimum of x-data.
            - x_max:float: Maximum of x-data.
            - y_min:float: Minimum of y-data.
            - y_max:float: Maximum of y-data.
        """

        # Set axes limits over whole range.
        if self.num_std==None:
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()

            if self.isBuffer:
                x_min *= 0.9 if x_min>0 else 1.1
                x_max *= 1.1 if x_max>0 else 0.9
                y_min *= 0.9 if y_min>0 else 1.1
                y_max *= 1.1 if y_max>0 else 0.9

            
        # Set axes limits over the concentrated range.
        else:
            x_mean = x.mean()
            x_std = x.std()

            x_min = x_mean - self.num_std*x_std
            x_max =  x_mean + self.num_std*x_std

            y_mean = y.mean()
            y_std = y.std()

            y_min = y_mean - self.num_std*y_std
            y_max = y_mean + self.num_std*y_std

            # Determines if the bounds are larger than the data.
            x_top = x.max()
            x_bottom = x.min()
            y_top = y.max()
            y_bottom = y.min()

            if x_max > x_top:
                x_max = x_top
            if x_min < x_bottom:
                x_min = x_bottom
            if y_max > y_top:
                y_max = y_top
            if y_min < y_bottom:
                y_min = y_bottom


        if x_min==x_max:
            x_min -= abs(x_min)
            x_max += abs(x_max)
        if y_min==y_max:
            y_min -= abs(y_min)
            y_max += abs(y_max)





        assert x_max>x_min and y_max>y_min, (f'Range problem for {x} or {y}: \n{x_min}>={x_max} or {y_min}>={y_max}')

        return x_min,x_max,y_min,y_max


