import os

import numpy as np
import matplotlib.pyplot as plt


from .dataRetrieval import *

from pathlib import Path
pathtohere = Path.cwd()

class DataVisualiser:
    """
    Stores functions to do with data visulisation. Includes the spectra analysis and field flow visulisation.

    plot_spectra
    plot_zoomedSpectrum
    flowField

    """

    def __init__(self, directoryName:str, variables:tuple):
        """
        Initalises the class.

        Inputs:
            - directoryName:str: Directory storing the files with the data.
            - variables:tuple: Variables to examine the correlation of.
        """

        self.directoryName = directoryName
        self.variables = variables
        self.numVariables = len(variables)

        __,self.data_ast = prepareData(self.directoryName, self.variables)

        try:
            os.mkdir(pathtohere / ('data/' + self.directoryName + '/dataAnalysis'))
        except:
            print("Directory 'dataAnalysis' already exists.")



    
   
    def plot_spectra(self,spectrumData, wmin:int=850, wmax:int=1850, smooth:int=21):
        """
        Display all spectra from .spec file.
        (`smooth' has not been implemented).

        Inputs:
            - spectrumData: Instances of the data as astropy dictionary.
            - wmin:int: Wavelength start [angstroms].
            - wmax:int: Wavelength end [angstroms].
            - smooth:int: Smoothing window [angstroms].
        """

        # Set figure properties.
        labels = spectrumData.keys()
        numVariables = len(labels)
        ncols = 4
        nrows = int(np.ceil(numVariables/ncols))
        
        fig = plt.figure(figsize=(12,12), dpi=200, tight_layout=True)


        for i in range(ncols):
            for j in range(nrows):
                index = i+ncols*j
                if index>=numVariables:
                    break
                ax = fig.add_subplot(nrows, ncols, index+1)
                ax.plot(spectrumData['Lambda'], spectrumData[labels[index]], label=labels[index])
            
                # Create appropiate labels
                if j==nrows-1 or True:
                    ax.set_xlabel(r'wavelength / $\lambda$ [A]')
                ax.set_ylabel(labels[index])


        plt.savefig(pathtohere / ('data/' + self.directoryName + '/dataAnalysis/spectrum_comparison.png'),
                        bbox_inches='tight')


    def plot_zoomedSpectrum(self,spectrumData, wmin:int, wmax:int, smooth:int=21):
        """
        Display zoomed in spectrum (i.e. on the P-Cygni profile).
        (`smooth' has not been implemented).

        Inputs:
            - spectrumData: Instances of the data as astropy dictionary.
            - wmin:int: Wavelength start [angstroms].
            - wmax:int: Wavelength end [angstroms].
            - smooth:int: Smoothing window [angstroms].
        """

        HeI = 5876
        wavelength = spectrumData['Lambda'] # This is in reverse order for some reason.

        include = np.where((wavelength>wmin) & (wavelength<wmax))

        wavelength = wavelength[include]
        intensity = spectrumData[spectrumData.keys()[-1]][include]


        fig = plt.figure(figsize=(10,6), dpi=200)
        ax = fig.add_subplot()

        # Draw zoomed spectrum.
        ax.plot(wavelength, intensity, c='b')
        ax.axvline(HeI, c='g')

        # Create appropiate labels.
        ax.set_xlabel(r'wavelength / $\lambda$ [A]')
        ax.set_ylabel('flux [$ergs\,cm^{-2}\,s^{-1}\,\AA^{-1}$]')
        ax.text(HeI+1, np.mean(intensity), 'HeI',c='g')


        plt.savefig(pathtohere / ('data/' + self.directoryName + '/dataAnalysis/zoomed_spectrum.png'),
                        bbox_inches='tight')



    def flowField(self, all_data_ast):
        """
        Creates a streamplot overlayed on an electron temperature plot of the vector field.

        Inputs:
            - all_data_ast: Data of the wind as an astropy dictionary.
        """

        notInclude = np.where((all_data_ast['inwind']!=0) | (all_data_ast['converge']!=0))
        sidelength = int(np.sqrt(len(all_data_ast['rho'])))
        
        R,Z = get_2d_positions(all_data_ast, sidelength)
        V_R = all_data_ast['v_x']
        
        V_Z = all_data_ast['v_z']
        
        # Only include inside of wind and converged.
        V_R[notInclude] = 0
        V_Z[notInclude] = 0

        V_R = V_R.reshape((sidelength,sidelength)).T
        V_Z = V_Z.reshape((sidelength,sidelength)).T


        
        all_data_ast['vxz'] = np.log10(1+all_data_ast['v_x']**2 + all_data_ast['v_z']**2)/2
        density2D = all_data_ast['rho'].reshape((sidelength,sidelength)).T[1:,1:]
        
        # Variable for colourmap.
        cmap_variable = 't_e'
        temperature1D = all_data_ast[cmap_variable]
        temperature1D[notInclude] = 0
 

        min_T = np.min(temperature1D)
        max_T = np.max(temperature1D)
        print('Temperature range before:',min_T,max_T)
        temperature2D = all_data_ast[cmap_variable].reshape((sidelength,sidelength)).T[1:,1:]

        # Formatting.
        axesRanges = (R.min(), R.max(), Z.min(), Z.max())
        ratio = 1.*(axesRanges[1] - axesRanges[0]) / (axesRanges[3] - axesRanges[2])


        fig = plt.figure(figsize=(8,8), dpi=300)
        ax = fig.add_subplot(111)

        # Plot vector field.
        ax.streamplot(R,Z,V_R,V_Z, color='lightblue', broken_streamlines=True,
                       density=1., linewidth=0.5, zorder=3)
    

        # Plot temperature distribution.
        print('Flowfield temperature range:',np.min(temperature2D),np.max(temperature2D))
        im_T = ax.imshow(temperature2D, cmap='hot',
                       origin='lower', extent=axesRanges,
                       aspect=ratio, interpolation='bicubic', zorder=1, vmin=0, vmax=max_T)
        cbar_T = plt.colorbar(im_T)


        # Create appropiate labels.
        ax.set_xlabel('log(r)')
        ax.set_ylabel('log(z)')
        cbar_T.ax.set_ylabel(cmap_variable)
        cbar_T.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        


        plt.savefig(pathtohere / ('data/' + self.directoryName + '/dataAnalysis/flowField.png'),
                bbox_inches='tight')





def get_2d_positions(all_data_ast, sidelength:int=None):
    """
    Get the 2d position coordinates for heatmaps.

    Inputs:
        - all_data_ast: All astropy data.
        - sidelength:int: Length of the side of the plot.

    Outputs:
        - R:np.ndarray: 2d radial positions.
        - Z:np.ndarray: 2d heights.

    """

    if sidelength==None:
        sidelength = int(np.sqrt(len(all_data_ast['xcen'])))

    R = np.log10(all_data_ast['xcen'].reshape((sidelength,sidelength)).T)
    Z = np.log10(all_data_ast['zcen'].reshape((sidelength,sidelength)).T)
    
    ## This part should not be needed in the future. 
    # See `Disc Winds Matter J. Matthews Ch 3.' for reason for odd sampling.
    gradient_R = (R[0,-1] - R[0,1]) / (sidelength-1)
    gradient_Z = (Z[-1,0] - Z[1,0]) / (sidelength-1)

    r = np.linspace(R[0,1] - gradient_R, R[0,-1], sidelength)
    z = np.linspace(Z[1,0] - gradient_Z, Z[-1,0], sidelength)

    R, Z = np.meshgrid(r,z)
    ####################

    return R,Z




def alt_log(x):
    """
    Continuous logarithm.

    Inputs:
        - x: Input data.

    Outputs:
        - y: Logged data.
    """


    if isinstance(x,float):
        isNotZero = True if x!=0 and x!=np.inf else False

    else:
        isNotZero = np.where((x!=0) & (x!=np.inf))

    y = x


    
    y[isNotZero] = x[isNotZero]/np.abs(x[isNotZero]) * np.log10(1 + np.abs(x[isNotZero]))

    return y


def logVariables(data_ast, variables:tuple, logged_var:set={}):
    """
    Log particular variables.

    Inputs:
        - data_ast: Astopy data.
        - variables:tuple: Variables that are available.
        - logged_var:set: Variables to be alt_logged.

    Outputs:
        - logged_data_ast: Logged astropy data.


    """

    logged_data_ast = dict.fromkeys(data_ast.keys())

    epsilon = 1e-15
    for i,var in enumerate(variables):  
        logged_data_ast[var] = data_ast[var]
        # Do not log data.
        if var not in logged_var:
            continue
        # Log data.
        logged_data_ast[var][np.where(data_ast[var]<epsilon)] = epsilon
        logged_data_ast[var] = np.log10(data_ast[var])

    return logged_data_ast






class PointComparer:
    """
    Class for storing functions relevant for generating plots comparing the points of two different .pf points.

    plot_pairs
    compositeComparePair
        _get_cmapPlottingInfo
        _createMiniPlots
        _createBigPlots
    """


    def __init__(self, data_ast0, data_ast1,name0:str,name1:str, logged_var:set={}):
        """
        Initalise the class.

        Inputs:
            - data_ast0: First astropy data.
            - data_ast1: Second astropy data.
            - name0:str: Name of the first point.
            - name1:str: Name of the second point.
            - logged_var:set: Variables to log.
        """

        self.data_ast = [data_ast0,data_ast1]
        self.names = (name0,name1)
        self.logged_var = logged_var

        for i in range(2):
            self.data_ast[i] = logVariables(self.data_ast[i],self.data_ast[i].keys(),self.logged_var)

        self.include = [(self.data_ast[i]['inwind']==0) & (self.data_ast[i]['converge']==0) for i in range(2)]


    
    def plot_pairs(self,variables:tuple):
        """
        Plot specific pairs of quantities for certain points in the .pf parameter space.

        Inputs:
            - variables:tuple: Variables available.
        """

   
        variablesToCompare = (('ne','t_e'),('xi','t_r'),('xi','h1'),('xi','he1'), ('h1','he1'), ('t_e','t_r'))

        numVarComp = len(variablesToCompare)

        
        fig = plt.figure(figsize=(8,16), tight_layout=True)
        for i in range(2):
            whereInclude = np.where(self.include[i])

            for j in range(numVarComp):
                ax = fig.add_subplot(numVarComp,2,i+2*j+1)
                # Display points with xcen as colourmap.
                ax.scatter(self.data_ast[i][variablesToCompare[j][0]][whereInclude],
                        self.data_ast[i][variablesToCompare[j][1]][whereInclude],
                        c=self.data_ast[i]['xcen'][whereInclude], marker='.', alpha=0.9)


                # Create appropiate labels.
                if j==0:
                    ax.set_title(self.names[i])
                if i==0:
                    ax.set_ylabel(variablesToCompare[j][1])
                ax.set_xlabel(variablesToCompare[j][0])

                # Set axes ranges.
                rangex,rangey = self._get_axesRanges(self.data_ast, *variablesToCompare[j],whereInclude)
                ax.set_xlim(rangex)
                ax.set_ylim(rangey)

        plt.savefig(pathtohere / ('plots/zoomed.png'), bbox_inches='tight')

    def compositeComparePair(self):
        """
        Plots a bunch of summary results consisting of how different variables change through space and with
        each other.
        """

        # Define variables to examine.
        self.columnVariables = ('t_e', 'ne', 'h1', 'he1') ## he2 -> he1 at some point
        self.cmaps = ('hot', 'Greens', 'Blues', 'Purples')

        nrows = len(self.columnVariables)

        assert nrows%2==0, 'Must have an even number of rows (preferably four).'
        assert len(self.columnVariables)==len(self.cmaps)


        compareVariables = ('ne','t_e')

        self.sidelength = int(np.sqrt(len(self.data_ast[0]['rho'])))
        
        self.R,self.Z = get_2d_positions(self.data_ast[0], self.sidelength)


        fig = plt.figure(figsize=(12,12), layout='tight', dpi=300)
        fig = self._createMiniPlots(fig, nrows)
        fig = self._createBigPlots(fig,'ne','t_e','xcen')
        
        
        plt.savefig(pathtohere / 'plots/mini-big.png', bbox_inches='tight')
    


    def _createMiniPlots(self, fig, nrows):
        """
        Create the mini-plots on the left hand side.

        Inputs:
            - fig: Instance of the figure.
            - nrows:int: Number of rows.

        Outputs:
            - fig: Altered figure.
        """



        for i in range(2):
            for j in range(nrows):
                ax = fig.add_subplot(nrows, 4, i+4*j+1)
                
                (var2D,minVar,maxVar,
                        axesRanges,ratio) = self._get_cmapPlottingInfo(self.data_ast[i],
                                                                        self.columnVariables[j],
                                                                        self.include[i])
                # Create mini-plot
                im = ax.imshow(var2D, cmap=self.cmaps[j],
                           origin='lower', extent=axesRanges,
                           aspect=ratio, interpolation=None,
                           zorder=1, vmin=minVar, vmax=maxVar)
                cbar = fig.colorbar(im,ax=ax)
                
                # Colour-in background.
                mask = np.asarray(~self.include[i].reshape((self.sidelength,self.sidelength)).T[1:,1:],float)
                ax.imshow(mask,cmap='binary',
                           origin='lower', extent=axesRanges,
                           aspect=ratio, interpolation=None,
                           zorder=2,alpha=mask)

                # Create appropiate labels.
                if j==0:
                    ax.set_title(self.names[i], fontsize=16)
                if i==0:
                    thisName = f'log({self.columnVariables[j]})' if self.columnVariables[j] in self.logged_var else self.columnVariables[j]
                    ax.set_ylabel(thisName, fontsize=16)

        return fig


    def _get_cmapPlottingInfo(self, data_ast_i, cmap_variable:str,include:np.ndarray):
        """
        Get relevant data and axes information for colourmap plotting.

        Inputs:
            - data_ast_i: Astropy data (could be one of multiple).
            - cmap_variable:str: The variable to plot.
            - include:np.ndarray: What points to include.

        Outputs:
            - var2D:np.ndarray: 2D colourmap points for the given variable.
            - minVar:float: Minimum value of the variable.
            - maxVar:float: Maximum value of the variable.
            - axesRanges:tuple: Ranges of the axes.
            - ratio:float: Ratio of the lengths of the axes.
        """

        var1D = data_ast_i[cmap_variable]

        
        var1D[np.where(~include)] = 0 if cmap_variable not in self.logged_var else -15

        minVar = np.min(var1D[np.where(include)])
        maxVar = np.max(var1D[np.where(include)])
        var2D = data_ast_i[cmap_variable].reshape((self.sidelength,self.sidelength)).T[1:,1:]
        print(f'Range of: {cmap_variable} ({minVar},{maxVar})')

        # Formatting.
        axesRanges = (self.R.min(), self.R.max(), self.Z.min(), self.Z.max())
        ratio = 1.*(axesRanges[1] - axesRanges[0]) / (axesRanges[3] - axesRanges[2])

        return var2D,minVar,maxVar,axesRanges,ratio


    
    def _createBigPlots(self,fig,xvar:str,yvar:str,cmapVar:str):
        """
        Create the big plots on the right hand side.

        Inputs:
            - fig: Instance of the figure.
            - xvar:str: x variable.
            - yvar:str: y variable.
            - cmapVar:str: colourmap variable.

        Outputs:
            - fig: Modified instance of the figure.
        """

        # Only include entries that are in the wind.
        data_ast_scat = list(self.data_ast)
        for i in range(2):
            # Just to make it clear, i is the pf index (0/1),
            # var is the variable and self.include[i] are points to include.
            data_ast_scat[i][xvar] = self.data_ast[i][xvar][self.include[i]]
            data_ast_scat[i][yvar] = self.data_ast[i][yvar][self.include[i]]
            data_ast_scat[i][cmapVar] = self.data_ast[i][cmapVar][self.include[i]]

        
        rangex,rangey = self._get_axesRanges(data_ast_scat, xvar,yvar)


        # Create big plots.
        for i in range(2):
            ax = fig.add_subplot(2,2,2*i+2)
            ax.scatter(self.data_ast[i][xvar], self.data_ast[i][yvar],
                    c=self.data_ast[i][cmapVar], marker='.', alpha=0.9)
            # Create approiate labels.
            name_x = f'log({xvar})' if xvar in self.logged_var else xvar
            name_y = f'log({yvar})' if yvar in self.logged_var else yvar
            ax.set_xlabel(name_x, fontsize=16)
            ax.set_ylabel(name_y, fontsize=16)
            ax.set_title(self.names[i], fontsize=16)
            
            # Set ranges.
            ax.set_xlim(rangex)
            ax.set_ylim(rangey)

        return fig

    @staticmethod
    def _get_axesRanges(data_ast_scat,xvar:str,yvar:str,whereInclude=True):
        """
        Get axes ranges.

        Inputs:
            - data_ast_scat: Astropy data.
            - xvar:str: x-variable.
            - yvar:str: y-variable.
        """

        #print(len(data_ast_scat[0][xvar]), len(data_ast_scat[1][xvar]))
        #print([np.min(data_ast_scat[i][xvar][whereInclude]) for i in range(2)])

        rangex = (min([np.min(data_ast_scat[i][xvar][whereInclude]) for i in range(2)]),
                  max([np.max(data_ast_scat[i][xvar][whereInclude]) for i in range(2)]))
        rangey = (min([np.min(data_ast_scat[i][yvar][whereInclude]) for i in range(2)]),
                  max([np.max(data_ast_scat[i][yvar][whereInclude]) for i in range(2)]))


        return rangex, rangey



    

    

    
