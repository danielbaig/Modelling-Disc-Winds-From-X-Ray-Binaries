#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <string>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <fstream>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>



class DataManager
{
    private:
        const unsigned int m_maxFileSize{ static_cast<unsigned int>(1e+4)};
        unsigned int m_maxIndex{};
        unsigned int m_numDataPoints{};
        unsigned int m_numCols{};

        const unsigned int m_inwindIndex{ 6 };
        const unsigned int m_convergeIndex{ 7 };

        const unsigned int m_numCols_pf{ 6 };
        const unsigned int m_numVariables_pf {4};
        std::vector<unsigned int> m_loggedCols_pf {1,2,3};


    public:
        DataManager();

        ~DataManager();

        void saveSamplePositions(Eigen::Tensor<double,4> samplePositions) const;

        void saveEnergyHistory(std::vector<double>* energyHistory_mean, std::vector<double>* energyHistory_std) const;

        void testIndicies(unsigned int fileIndex, std::string line,
                          unsigned int colnum, unsigned int rownum) const;

        void testIndicies(std::string line, unsigned int colnum, unsigned int rownum) const;

        Eigen::Tensor<double,3> get_potentials();

        void normalise(Eigen::MatrixXd* potentialPositions_pf);

        void read_potentialFile_pf(Eigen::MatrixXd* potentialPositions_pf, std::ifstream* myfile);        

        Eigen::MatrixXd get_potentials_pf();

        std::vector<double> get_densities();
};













#endif
