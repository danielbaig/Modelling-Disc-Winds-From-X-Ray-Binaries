#ifndef MC_MINIMISER_H
#define MC_MINIMISER_H

#include "Energy.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>



class MC_minimiser
{
    private:
        std::vector<double> m_std {1e-1,2e-1,3e-1,5e-2,1e-2,2e-1}; // fragile
        const double m_boltzTemperature {1e+15*1.38e-23};
        Eigen::VectorXd m_energyHistory {};
        unsigned int m_energyPeriod {};

        // Constructor variables.
        Eigen::MatrixXd m_samplePositions {};
        const unsigned int m_numSamples {};
        const unsigned int m_numDims {};
        Energy m_energy;
        const unsigned int m_numSteps {};
        


    public:
        MC_minimiser(Eigen::MatrixXd samplePositions, Energy energy,
                        const unsigned int numSteps, const unsigned int energyPeriod);

        ~MC_minimiser();

        std::tuple<Eigen::Tensor<double,3>, Eigen::VectorXd> runMC();

        Eigen::VectorXd get_movement();

};


#endif
