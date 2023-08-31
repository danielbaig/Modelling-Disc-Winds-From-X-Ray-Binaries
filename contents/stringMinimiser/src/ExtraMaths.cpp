#include "ExtraMaths.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <cassert>


double sumSquare(const double& x, const double& y) { return x + y*y; }

double approxSigmoid(const double x) 
{ 
    /*
    Approximate sigmoid function.

    Inputs:
        - const double x: Input (>=0).

    Outputs:
        - double y: Sigmoid evaluated at x.

    */
    // More computationally effective if only using x>0 since do not need to take abs().
    assert(x>=0);
    return x / (1 + x); 
}


