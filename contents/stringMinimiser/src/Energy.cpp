#include "Energy.h"
#include "ExtraMaths.h"

Energy::Energy(Eigen::MatrixXd samplePositions,
               const Eigen::MatrixXd potentialPositions,
               const Eigen::MatrixXd distances_pf,
               const double relaxedLength, std::vector<double>* densities) 
        : m_potentialPositions {potentialPositions},
          m_distances_pf {distances_pf},
          m_numSamples {static_cast<unsigned int>(samplePositions.rows())},
          m_numPotentials {static_cast<unsigned int>(potentialPositions.rows())},
          m_numDims {static_cast<unsigned int>(samplePositions.cols())},
          m_relaxedLength {relaxedLength},
          m_densities {densities}

{
    /*
    Constructor for the class.

    Inputs:
        - Eigen::MatrixXd samplePositions: Positions of the samples in vps.
        - const Eigen::MatrixXD potentialPositions: Positions of the potentials in vps.
        - const Eigen::MatrixXd distances_pf: Distance ratios in pfps.
        - const double relaxedLength: Relaxed length of sprin inbetween nodes.
        - std::vector<double>* densities: Local density of points in pfps.
    */
    
    m_energy = calculateTotalEnergy(samplePositions);
}

Energy::~Energy()
{
    return;
}

double Energy::calculateElasticEnergy(const Eigen::VectorXd samplePos1, const Eigen::VectorXd samplePos2)
{
    /*
    Calculate the elastic potential energy between two adjacent nodes in the string.

        Inputs:
            - const Eigen::VectorXd samplePos1: First node position.
            - const Eigen::VectorXd samplePos2: Second node position.
        
        Outputs:
            - double elasticPotentialEnergy: Calculated energy.
    */

    double extention {(samplePos1 - samplePos2).norm() - m_relaxedLength};
    return extention*extention;
}

double Energy::calculateFieldEnergy(const Eigen::VectorXd r_point,
                            const Eigen::VectorXd r_potential,
                            const double distance_pf,
                            const double density)
{
    /*
    Gets the potential due to a single source as seen from a single point in pf parameter space.

    Inputs:
        - const Eigen::VectorXd r_point: Sample point's position in vps.
        - const Eigen::VectorXd r_potential: Potential source's position in vps.
        - const double distance_pf: Ratio between the distance between the two points
                               in pfps to the distance between the start and
                               end of the sample line.
        - const double density: Local density of the field point.

    Outputs:
        - double potential: The potential for this sample point.
    */

    const Eigen::VectorXd dr_vec{r_point - r_potential};
    const double dr {dr_vec.norm()};
    
    // return distance_pf * dr / density;
    return approxSigmoid(distance_pf) * approxSigmoid(dr) * (1 - approxSigmoid(density));
    
}

double Energy::calculateTotalEnergy(Eigen::MatrixXd samplePositions)
{
    /*
    Calculate the energy of the system.

    Inputs:
        - Eigen::MatrixXD samplePositions: Positions of the samples in vps.

    Outputs:
        - double energy: Energy of the system.
    */
    assert(m_numSamples%2==1);

    double energy { 0 };
    for (unsigned int i { 0 }; i < m_numSamples; ++i)
    {
        // To change to prevent double counting of the spring energy.
        if (i != 0 && i != m_numSamples-1/* && i%2==1 */)
            {
                energy += calculatePointEnergy(samplePositions.row(i-1), samplePositions.row(i),
                                                samplePositions.row(i+1), i);
            }
    }
    return energy;
}

double Energy::calculatePointEnergy(Eigen::VectorXd leftPosition, Eigen::VectorXd thisPosition,
                                    Eigen::VectorXd rightPosition, unsigned int this_i)
{
    double energy {};

    // Spring energy.
    energy += calculateElasticEnergy(thisPosition, leftPosition);
    energy += calculateElasticEnergy(thisPosition, rightPosition);

    energy *= m_halfSpringEnergyRatio;


    
    // Field energy.
    for (unsigned int j { 0 }; j < m_numPotentials; ++j)
    {

        energy += calculateFieldEnergy(thisPosition,
                                              m_potentialPositions.row(j),
                                              m_distances_pf(this_i,j),
                                              (*m_densities)[j]);
    }
    return energy;

}

double Energy::get_energy() {return m_energy;}


