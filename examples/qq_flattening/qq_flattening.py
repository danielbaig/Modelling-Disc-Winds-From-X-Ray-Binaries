import numpy as np
from scipy.stats import skewnorm, cauchy


def generateData():
    """
    Generate data that is used to visulise the common features in linearlised quantile-quantile plots.

    Outputs:
        - distributionNames:tuple: Names of the distributions.
        - data0:np.ndarray: Normal distribution to compare to.
        - data1:np.ndarray: Distribution data.
    """


    distributionNames = ('identical', 'heavy tailed', 'light tailed', 'bimodal',
                        'left skew', 'right skew', 'spread', 'tight')
    numDistributions = len(distributionNames)

    rng = np.random.default_rng()
    numPoints = int(1e+3)

    data1 = np.empty((numDistributions, numPoints))
    
    lightTail_height = 1. # 0.4
    lightTailed = lambda x: (-2*np.log(x/lightTail_height))**(1/4)

    # Testing distribution differences from:
    # <https://stats.stackexchange.com/questions/101274/how-to-interpret-a-qq-plot>

    # Generate distributions.
    data1[0] = rng.normal(0,1,numPoints)
    data1[1] = cauchy.rvs(0,0.1,numPoints)
    data1[2] = lightTailed(rng.uniform(0,lightTail_height, numPoints))
    data1[3] = np.concatenate((rng.normal(-2,1,numPoints//2), rng.normal(2,1,numPoints//2)))
    data1[4] = skewnorm.rvs(1,size=numPoints)
    data1[5] = skewnorm.rvs(-1,size=numPoints)
    data1[6] = rng.normal(0,2,numPoints)
    data1[7] = rng.normal(0,0.5,numPoints)

    data0 = rng.normal(0,1,(numDistributions,numPoints))
    
 

    return distributionNames, data0,data1

