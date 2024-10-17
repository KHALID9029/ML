#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <Eigen/Core>

										/// LinearRegression class definition ///

class LinearRegression {
public:
    void fit( std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels);
    std::vector<double> predict( std::vector<std::vector<double>>& testData);

    // Overloaded fit function: Gradient Descent
    void fit( std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, double learningRate, int numEpochs);
    
    // Overloaded predict function: Gradient Descent version (same as before, using stored coefficients)
    std::vector<double> predictGradient( std::vector<std::vector<double>>& testData);

    std::tuple<double, double, double, double, double, double,
        std::vector<double>, std::vector<double>,
        std::vector<double>, std::vector<double>>
        runLinearRegression(const std::string& filePath, int trainingRatio);

private:

    Eigen::VectorXd m_coefficients; // Store the coefficients for future predictions
    std::vector<double> coefficients; // Store the coefficients

};

#endif // LINEARREGRESSION_H
