#include "DataManager.h"

#include <iomanip>
#include <iostream>



DataManager::DataManager()
{
    return;
}

DataManager::~DataManager()
{
    return;
}


void DataManager::saveSamplePositions(Eigen::Tensor<double,4> samplePositions) const
{
    /*
    Saves the sample positions to a .txt file.

    Inputs:
        - Eigen::MatrixXd samplePositions: Positions of the samples.
    */


    std::ofstream myfile;
    // Dimension data.
    myfile.open("../data/samplePositionsDims.txt");
    std::cout << "Saving rank-4 tensor of size: (";
    for (unsigned int i{ 0 }; i < 4; ++i)
    {
        myfile << std::setprecision(15) << samplePositions.dimension(i) << '\n';
        std::cout << samplePositions.dimension(i) << ',';
    }

    std::cout << ")\n";
    myfile.close();


    // Write datafile.
    myfile.open("../data/samplePositions.txt");

    for (int i{ 0 }; i < samplePositions.dimension(0); ++i)
    {
        for (unsigned int j{ 0 }; j < samplePositions.dimension(1); ++j)
        {
            for (unsigned int k{ 0 }; k < samplePositions.dimension(2); ++k)
            {
                for (unsigned int l{ 0 }; l < samplePositions.dimension(3); ++l)
                {
                    myfile << std::setprecision(15) << samplePositions(i,j,k,l) << ',';
                }
            }
        }
    }

    myfile.close();
}


void DataManager::saveEnergyHistory(std::vector<double>* energyHistory_mean, std::vector<double>* energyHistory_std) const
{
    /*
    Saves the history of the energy to a .txt file.

    Inputs:
        - Eigen::VectorXd energyHistory_mean: Mean of the energy at each step.
        - Eigen::VectorXd energyHistory_std: Standard deviation of the energy at each step.
    */

    std::ofstream myfile;
    myfile.open("../data/energyHistory.txt");

    for (int i{0}; i < energyHistory_mean->size(); ++i)
    {
        myfile << i << ',' << std::setprecision(15) << (*energyHistory_mean)[i] << ','
                    << std::setprecision(15) << (*energyHistory_std)[i] << '\n';
    }
    myfile.close();


}


void DataManager::testIndicies(unsigned int fileIndex, std::string line,
                  unsigned int colnum, unsigned int rownum) const
{
    /*
    Tests that the indicies are in range for the vps tensor.

    Inputs:
        - unsigned int fileIndex: pf sample index.
        - std::string line: Line being read.
        - unsigned colnum: Column number.
        - unsigned rownum: Row number.
    */

    if (fileIndex > m_maxIndex || rownum > m_maxFileSize-1 || colnum > m_numCols)
    {
        std::cout << "Error: Tensor indicies are greater than the tensor size:\n";
        std::cout << "Indicies: " << fileIndex << ' ' << rownum << ' ' << colnum << '\n';
        std::cout << "Value: " << line << '\n';
        std::cout << "Limits: " << m_maxIndex << ' ' 
                  << m_maxFileSize-1 << ' ' << m_numCols << '\n';
    }

}

void DataManager::testIndicies(std::string line, unsigned int colnum, unsigned int rownum) const
{
    /*
    Tests that the indicies are in range for the pfps tensor.

    Inputs:
        - std::string line: Line being read.
        - unsigned colnum: Column number.
        - unsigned rownum: Row number.
    */

    if (rownum > m_maxFileSize-1 || colnum > m_numCols)
    {
        std::cout << "Error: Matrix indicies are greater than matrix size:\n";
        std::cout << "Indicies: " << rownum << ' ' << colnum << '\n';
        std::cout << "Value: " << line << '\n';
        std::cout << "Limits: " << m_maxIndex << ' '
                  << m_maxFileSize-1 << ' ' << m_numCols << '\n';
    }

}

Eigen::Tensor<double,3> DataManager::get_potentials()
{
    /*
    Gets the potential positions in the vps from a .txt file.

    Outputs:
        - Eigen::Tensor<double,3> potentialPositions: Positions of each of the potentials in vps.
    */

    std::ifstream myfile{};
    myfile.open("../data/potentialPositionsDims.txt");

    if (!myfile)
    {
        std::cout << "Error: unable to read file: potentialPositionsDims.txt\n";
        return {};
    }

    // Gets dimension sizes.
    std::string line{};
    myfile >> line;
    m_maxIndex = std::stod(line)-1;
    myfile >> line;
    m_numDataPoints = std::stod(line);
    myfile >> line;
    m_numCols = std::stod(line);


    myfile.close();
    myfile.open("../data/potentialPositions.txt");

    if (!myfile)
    {
        std::cout << "Error: unable to read file: potentialPositions.txt\n" ;
        return {};
    }

    Eigen::Tensor<double,3> potentialPositions(m_maxIndex+1, m_numDataPoints, m_numCols);
    std::cout << "Getting tensor of size: (" <<m_maxIndex+1 << ", " << m_numDataPoints 
              << ", " << m_numCols << ")\n";

    for (int i{ 0 }; i < m_maxIndex+1; ++i)
    {
        for (int j{ 0 }; j < m_numDataPoints; ++j)
        {
            for (int k{ 0 }; k < m_numCols; ++k)
            {
                myfile >> line;
                potentialPositions(i,j,k) = std::stod(line);
            }
        }
    }
    myfile.close();

    return potentialPositions;
}

void DataManager::normalise(Eigen::MatrixXd* potentialPositions_pf)
{
    /*
    Normalise the potential positions in pfps.

    Inputs:
        - Eigen::MatrixXd* potentialPositions_pf: Positions of the potentials in pfps.
    */

    Eigen::VectorXd minimums {potentialPositions_pf->colwise().minCoeff()};
    Eigen::VectorXd maximums {potentialPositions_pf->colwise().maxCoeff()};


    for (int i{ 0 }; i < potentialPositions_pf->rows(); ++i)
    {
        for (int j{ 0 }; j < potentialPositions_pf->cols(); ++j)
        {
            if (maximums(j)==minimums(j))
            {
                (*potentialPositions_pf)(i,j) = 1.;
                continue;
            }

            (*potentialPositions_pf)(i,j) = ((*potentialPositions_pf)(i,j) - minimums(j))
                                             / (maximums(j) - minimums(j));
        }
    }
}






void DataManager::read_potentialFile_pf(Eigen::MatrixXd* potentialPositions_pf, std::ifstream* myfile)
{
    /*
    Read the data from the .txt filee to obtain the pfps potential positions.

    Inputs:
        - Eigen::MatrixXd* potentialPositions_pf: Matrix to hold the positions of the potentials in pfps.
        - std::ifstream* myfile: File to read from.
    */

    // Loop variables.
    std::string line {};
    *myfile >> line; // To get rid of the initial ###
    unsigned int rownum{ 0 };

    while (!myfile->eof() && rownum<m_maxIndex+2)
    {

        for (unsigned int colnum{ 0 }; colnum < m_numCols_pf; ++colnum)
        {
            *myfile >> line;
            
            if (rownum==0 || std::size(line)==0)
            {
                continue;
            }

            if (colnum!=0 && colnum<m_numVariables_pf+1)
            {
                testIndicies(line, colnum-1, rownum-1);

                (*potentialPositions_pf)(rownum-1,colnum-1) = std::find(m_loggedCols_pf.begin(),
                                        m_loggedCols_pf.end(), colnum)==m_loggedCols_pf.end()
                                    ? std::stod(line) 
                                    : std::log10(std::stod(line));
            }
        }
        ++rownum;
    }  
}


Eigen::MatrixXd DataManager::get_potentials_pf()
{
    /*
    Gets the positions of the potentials in pfps from a .txt file.

    Outputs:
        - Eigen::MatrixXd potentialPositions_pf: Positions of the potentials in pfps.
    */
    Eigen::MatrixXd potentialPositions_pf(m_maxIndex+1, m_numVariables_pf);

    std::ifstream myfile{};
    myfile.open("../../../data/features.txt");

    if (!myfile)
    {
        std::cout << "Error: unable to read features.txt file.\n";
    }
    
    read_potentialFile_pf(&potentialPositions_pf, &myfile);
    normalise(&potentialPositions_pf);

    myfile.close();

    return potentialPositions_pf;
}


std::vector<double> DataManager::get_densities()
{
    /*
    Gets the densities around each pf point.

    Outputs:
        - std::vector<double> densities: Densities around each pf point.
    */

    std::vector<double> densities(m_maxIndex+1);

    std::ifstream myfile{};
    myfile.open("../data/densities.txt");

    std::string line{};
    unsigned int rownum{0};

    while (!myfile.eof() && rownum<m_maxIndex+1)
    {
        myfile >> line;
        densities[rownum] = std::stod(line);
        ++rownum;
    }

    myfile.close();


    return densities;

}


