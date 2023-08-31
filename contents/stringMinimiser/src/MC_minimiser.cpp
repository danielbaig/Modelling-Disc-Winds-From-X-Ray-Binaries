#include "MC_minimiser.h"
#include "Random.h"

#include <iostream>
#include <iomanip>
#include <numeric>

MC_minimiser::MC_minimiser(Eigen::MatrixXd samplePositions, Energy energy,
                            const unsigned int numSteps, const unsigned int energyPeriod)
        : m_samplePositions {samplePositions},
        m_numSamples {static_cast<unsigned int>(samplePositions.rows())},
        m_numDims {static_cast<unsigned int>(samplePositions.cols())},
        m_energy {energy},
        m_numSteps {numSteps},
        m_energyPeriod {energyPeriod}
{
    m_energyHistory.resize(m_numSteps + 1);
    m_energyHistory(0) = energy.get_energy();
}

MC_minimiser::~MC_minimiser()
{
    return;
}

    

std::tuple<Eigen::Tensor<double,3>, Eigen::VectorXd> MC_minimiser::runMC()
{
    /*
    Run the main Monte Carlo procedure.

    Outputs:
        - Eigen::MatrixXd m_samplePositions: New sample positions.
    */


    // Define loop variables.
    unsigned int pointToMove {};
    // Eigen::VectorXd dy(m_numDims);

    Eigen::VectorXd prev_pos(m_numDims);
    Eigen::VectorXd next_pos(m_numDims);
    int movingDim{};

    double energyChange {};
    double prev_energy {};
    double next_energy {};

    unsigned int searchingExp{0};
    Eigen::Tensor<double,3> all_samplePositions(m_numSteps / m_energyPeriod+1,m_numSamples,m_numDims);


    for (unsigned int i { 0 }; i < m_numSteps; ++i)
    {

        pointToMove = Random::intUniform(1, m_numSamples-1);
        

        prev_pos = m_samplePositions.row(pointToMove);

        // Non-plane constraining, but mire computationally intensive.
        // dy = get_movement();
        // next_pos = prev_pos + dy;

        // Plane constraining.
        next_pos = prev_pos;
        movingDim = Random::intUniform(0,m_numDims);
        next_pos(movingDim) += 1e-3*m_std[movingDim]*Random::randn();

        // Calculate local energy contribution.
        prev_energy = m_energy.calculatePointEnergy(m_samplePositions.row(pointToMove-1), prev_pos,
                                                     m_samplePositions.row(pointToMove+1),
                                                     pointToMove);

        next_energy = m_energy.calculatePointEnergy(m_samplePositions.row(pointToMove-1), next_pos,
                                                     m_samplePositions.row(pointToMove+1),
                                                     pointToMove);
        // std::cout << "Prev: " << prev_energy << "\tNext: " << next_energy << '\n';
        energyChange = next_energy - prev_energy;

        if (energyChange < 0 || std::exp(-energyChange / m_boltzTemperature) > Random::uniform())
        {
            m_samplePositions.row(pointToMove) = next_pos;

            m_energyHistory(i+1) = m_energyHistory(i) + energyChange;
        }
        else
        {
            m_energyHistory(i+1) = m_energyHistory(i);
        }



        // Save at periodic increments.
        if ((i+1) % m_energyPeriod==0 || i==0)
        {
            for (int k{ 0 }; k<m_numSamples; ++k)
            {
                for (int l{ 0 }; l<m_numDims; ++l)
                {
                    all_samplePositions(searchingExp,k,l) = m_samplePositions(k,l);
                }
            }


            ++searchingExp;
        }
    }


    return {all_samplePositions, m_energyHistory};


}

Eigen::VectorXd MC_minimiser::get_movement()
{
    Eigen::VectorXd dy(m_numDims);


    for (unsigned int i {0}; i < m_numDims; ++i)
    {
        dy(i) = 1e-3*m_std[i]*Random::randn() / m_numDims;
    }

    return dy;

}

