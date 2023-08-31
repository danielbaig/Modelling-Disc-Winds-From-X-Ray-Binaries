#include "ExtraMaths.h"
#include "Random.h"
#include "Energy.h"
#include "MC_minimiser.h"
#include "DataManager.h"
#include "Timer.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <fstream>
#include <string>

#include <unsupported/Eigen/CXX11/Tensor>
#include "TensorConvert.cpp"




std::tuple<Eigen::MatrixXd, double> setInitialPositions(unsigned int numSamples,
                                        Eigen::VectorXd startPosition,
                                        Eigen::VectorXd endPosition)
{
    /*
    Sets the initial positions of the sample points in vps and pfps. (Essentially np.linspace.)

    Inputs:
        - unsigned int numSamples: The number of samples.
        - std::vector<double> startPosition: The position of the first sample.
        - std::vector<double> endPosition: The position of the last sample.

    Outputs:
        - std::vector<std::vector<double>> initialPositions: The initial positions of the sample points.
        - double magnitude(dy): Relaxed length of the spring.
    */

    // Get step vector.
    Eigen::VectorXd dy {endPosition - startPosition};
    dy = dy / (numSamples-1);

    Eigen::MatrixXd initialPositions (numSamples, static_cast<int>(dy.size()));

    for (unsigned int i{0}; i<numSamples; ++i)
    {
        initialPositions.row(i) = i*dy + startPosition;
    }

    return {initialPositions, dy.norm()};
}


Eigen::MatrixXd get_distances_pf(const Eigen::MatrixXd potentialPositions_pf,
                                const Eigen::MatrixXd samplePositions_pf)
{
    /*
    Obtain the distance ratios between all samples and all points in the pfps against the distance
    between the path potential points.

    Inputs:
        - const Eigen::MatrixXd potentialPositions_pf: Positions of the potentials in the pfps.
        - const Eigen::MatrixXd samplePositions_pf: Positions of the samples in the pfps.

    Outputs:
        - Eigen::MatrixXd distances_pf: Ratios calculated between all points and potentials in pfps.
    */

    const int numSamples {static_cast<int>(samplePositions_pf.rows())};
    const int numPotentials {static_cast<int>(potentialPositions_pf.rows())};

    Eigen::MatrixXd distances_pf(numSamples, numPotentials);


    for (int i{0}; i<numSamples; ++i)
    {
        for (int j{0}; j<numPotentials; ++j)
        {
            distances_pf(i,j) = (potentialPositions_pf.row(j) - samplePositions_pf.row(i)).norm();
        }
    }

    return distances_pf;
}




std::tuple<Eigen::Tensor<double,4>, Eigen::MatrixXd> runPoints(Eigen::Tensor<double,3>* potentialPos,
                                                               Eigen::MatrixXd* potentialPositions_pf,
                                                               std::vector<double>* densities)
{
    constexpr unsigned int numSamples {11};
    constexpr int startPotential {0};
    constexpr int endPotential {2};
    constexpr unsigned int MC_steps(static_cast<int>(1e+4)); // >1e+5 recommended
    const unsigned int energyPeriod {static_cast<int>(MC_steps/100)};
    const unsigned int numEnergySamples{MC_steps / energyPeriod+1};
    const unsigned int progressPeriod{ static_cast<unsigned int>(potentialPos->dimension(1) / 10) };


    Eigen::MatrixXd energyHistories(potentialPos->dimension(1),MC_steps+1);
    Eigen::Tensor<double,4> all_samplePositions(potentialPos->dimension(1),numEnergySamples,
                                                numSamples,potentialPos->dimension(2));
    

    Eigen::MatrixXd samplePositions_pf {};
    Eigen::MatrixXd samplePositions {};
    double relaxedLength {};
    Eigen::MatrixXd distances_pf {};
    Eigen::VectorXd energyHistory{};
    Eigen::Tensor<double,3> sampledSamplePositions(numEnergySamples, numSamples, potentialPos->dimension(2));


    Timer t{};

    for (int p{ 0 }; p < potentialPos->dimension(1); ++p)
    {
        // Display progress. (should be replaced by tqdm at some point)
        if (p % progressPeriod==0)
        {
            std::cout << std::setprecision(3) << 100*static_cast<double>(p) / potentialPos->dimension(1) << "%\n";
        }

        Eigen::MatrixXd potentialPositions(MatrixCast(potentialPos->chip(p,1),
                                    potentialPos->dimension(0),potentialPos->dimension(2)));
        

        // pfps details. relaxedLength here is overwritten immediatly.
        std::tie(samplePositions_pf, relaxedLength) =
            setInitialPositions(numSamples,
                                static_cast<Eigen::VectorXd>(potentialPositions_pf->row(startPotential)),
                                static_cast<Eigen::VectorXd>(potentialPositions_pf->row(endPotential)));
        distances_pf = get_distances_pf(*potentialPositions_pf, samplePositions_pf);

        // vps details.
        std::tie(samplePositions, relaxedLength) =
            setInitialPositions(numSamples, static_cast<Eigen::VectorXd>(potentialPositions.row(startPotential)),
                                    static_cast<Eigen::VectorXd>(potentialPositions.row(endPotential)) );




        Energy energy {samplePositions, potentialPositions, distances_pf, relaxedLength, densities};

        // Monte Carlo procedure.
        MC_minimiser minimiser {samplePositions, energy, MC_steps, energyPeriod};
        
        std::tie(sampledSamplePositions, energyHistory) = minimiser.runMC();
        


        energyHistories.row(p) = energyHistory;
        
        // It may be possible to make this more efficient in the future. (map perhaps?)
        for (int i{ 0 }; i < numEnergySamples; ++i)
        {
            for (int j{ 0 }; j < numSamples; ++j)
            {
                for (int k{ 0 }; k < potentialPos->dimension(2); ++k)
                {
                    all_samplePositions(p,i,j,k) = sampledSamplePositions(i,j,k);
                }
            }
        }
    }

    std::cout << "Time taken: " << static_cast<double>(t.elapsed()) / 60. << " minutes\n";

    return {all_samplePositions, energyHistories};

}

std::tuple< std::vector<double>, std::vector<double> > get_averageEnergy(Eigen::MatrixXd* energyHistories)
{
    /*
    Gets the mean and standard deviation of each of the strings at each MC step.

    Inputs:
        - Eigen::MatrixXd* energyHistories: History of the energy of each of the strings.

    Outputs:
        - std::vector<double> energyHistory_mean: Mean energy at each MC step.
        - std::vector<double> energyHistory_std: Std of the energy at each MC step.
    */

    double numPoints{ static_cast<double>(energyHistories->rows()) };
    double numSteps{ static_cast<double>(energyHistories->cols()) };



    std::vector<double> energyHistory_mean(numSteps);
    std::vector<double> energyHistory_std(numSteps);
    double delta{};


    for (int step{ 0 }; step<numSteps; ++step)
    {
        for (int p{ 0 }; p<numPoints; ++p)
        {
            energyHistory_mean[step] += (*energyHistories)(p,step);
            energyHistory_std[step] += (*energyHistories)(p,step) * (*energyHistories)(p,step);
        }
        energyHistory_mean[step] /= numPoints;
        energyHistory_std[step] /= numPoints;
        energyHistory_std[step] -= energyHistory_mean[step]*energyHistory_mean[step];
        energyHistory_std[step] = std::sqrt(energyHistory_std[step]);

    }


    return {energyHistory_mean, energyHistory_std};
}


int main()
{
    std::cout << "START\n";

    // Get data.
    DataManager dataManager{};
    Eigen::Tensor<double,3> potentialPos {dataManager.get_potentials()};
    Eigen::MatrixXd potentialPositions_pf {dataManager.get_potentials_pf()};
    std::vector<double> densities {dataManager.get_densities()};


    auto [samplePositions, energyHistories] {runPoints(&potentialPos, &potentialPositions_pf, &densities)};

    auto [energyHistory_mean, energyHistory_std] {get_averageEnergy(&energyHistories)};

    

    // Save data.
    dataManager.saveSamplePositions(samplePositions);
    dataManager.saveEnergyHistory(&energyHistory_mean, &energyHistory_std);





	return 0;
}
