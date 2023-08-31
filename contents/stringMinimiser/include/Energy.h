#ifndef ENERGY_H
#define ENERGY_H

#include <Eigen/Dense>



class Energy
{
    /*
    Class for calculating the energy of the system and contributions due to different parts of the string.
    */
    private:
        const double m_halfSpringEnergyRatio{ 1e+4 };
        double m_energy{};
        

        // Constructor definitions.
        const Eigen::MatrixXd m_potentialPositions {};
        const Eigen::MatrixXd m_distances_pf {};
        const unsigned int m_numSamples {};
        const unsigned int m_numPotentials {};
        const unsigned int m_numDims {};
        const double m_relaxedLength {};
        std::vector<double>* m_densities{};




    public:
        Energy(Eigen::MatrixXd samplePositions,
               const Eigen::MatrixXd potentialPositions,
               const Eigen::MatrixXd distances_pf,
               const double relaxedLength, std::vector<double>* densities);

        ~Energy();

    
        double calculateElasticEnergy(const Eigen::VectorXd samplePos1, const Eigen::VectorXd samplePos2);

        double calculateFieldEnergy(const Eigen::VectorXd r_point,
                            const Eigen::VectorXd r_potential,
                            const double distance_pf,
                            const double density);

        double calculateTotalEnergy(Eigen::MatrixXd samplePositions);

        double calculatePointEnergy(Eigen::VectorXd leftPosition, Eigen::VectorXd thisPosition,
                                    Eigen::VectorXd rightPosition, unsigned int this_i);
        double get_energy();
};

#endif

