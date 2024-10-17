#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox

using namespace std;

///  LinearRegression class implementation  ///




// Function to fit the linear regression model to the training data //
void LinearRegression::fit(std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) 
{

    // This implementation is using Matrix Form method
    /* Implement the following:
        --- Check if the sizes of trainData and trainLabels match
        --- Convert trainData to matrix representation
        --- Construct the design matrix X
        --- Convert trainLabels to matrix representation
        --- Calculate the coefficients using the least squares method
        --- Store the coefficients for future predictions
    */

    // TODO


    if (trainData.size() != trainLabels.size()) 
    {
		MessageBox::Show("The sizes of trainData and trainLabels do not match.");
		return;
    }

    int numRows = trainData.size();
    int numCols = trainData[0].size();
    Eigen::MatrixXd X(numRows, numCols + 1); // +1 for bias term

    for (int i = 0; i < numRows; ++i) 
    {
        X(i, 0) = 1.0; // Bias term
        for (int j = 0; j < numCols; ++j) 
        {
            X(i, j + 1) = trainData[i][j];
        }
    }

    Eigen::VectorXd y(numRows); // Convert trainLabels to an eigen vector
    for (int i = 0; i < numRows; ++i) 
    {
        y(i) = trainLabels[i];
    }


    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd theta = XtX.inverse() * X.transpose() * y;

    m_coefficients = theta;
}



void LinearRegression::fit(std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, double learningRate, int numEpochs) 
{

    DataPreprocessor::normalizeDataset(trainData);


    int numRows = trainData.size();
    int numCols = trainData[0].size();
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(numCols + 1); // Coefficients including bias

    Eigen::MatrixXd X(numRows, numCols + 1); // Additional column for bias
    for (int i = 0; i < numRows; ++i) 
    {
        X(i, 0) = 1.0; // Bias term
        for (int j = 0; j < numCols; ++j) 
        {
            X(i, j + 1) = trainData[i][j];
        }
    }

    Eigen::VectorXd y(numRows);
    for (int i = 0; i < numRows; ++i) 
    {
        y(i) = trainLabels[i];
    }

    // Gradient Descent
    for (int epoch = 0; epoch < numEpochs; ++epoch) 
    {
        Eigen::VectorXd predictions = X * theta;

        Eigen::VectorXd gradient = (X.transpose() * (predictions - y)) / numRows;

        theta = theta - learningRate * gradient;
    }

    m_coefficients = theta;
}


// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(std::vector<std::vector<double>>& testData) 
{

    // This implementation is using Matrix Form method    
    /* Implement the following
        --- Check if the model has been fitted
        --- Convert testData to matrix representation
        --- Construct the design matrix X
        --- Make predictions using the stored coefficients
        --- Convert predictions to a vector
    */

    // TODO


    if (m_coefficients.size() == 0) 
    {
        throw std::runtime_error("Model has not been fitted.");
    }

    int numRows = testData.size();
    int numCols = testData[0].size();

    Eigen::MatrixXd X(numRows, numCols + 1); // Add a column for bias

    for (int i = 0; i < numRows; ++i) 
    {
        X(i, 0) = 1.0; // Bias term
        for (int j = 0; j < numCols; ++j) 
        {
            X(i, j + 1) = testData[i][j];
        }
    }

    Eigen::VectorXd predictions = X * m_coefficients;

    std::vector<double> result(numRows);
    for (int i = 0; i < numRows; ++i) 
    {
        result[i] = predictions(i);
    }

    return result;
}





// Function to make predictions on new data //
std::vector<double> LinearRegression::predictGradient(std::vector<std::vector<double>>& testData) 
{

    // This implementation is using Matrix Form method    
    /* Implement the following
        --- Check if the model has been fitted
        --- Convert testData to matrix representation
        --- Construct the design matrix X
        --- Make predictions using the stored coefficients
        --- Convert predictions to a vector
    */

    // TODO


    DataPreprocessor::normalizeDataset(testData);
    if (m_coefficients.size() == 0) 
    {
        throw std::runtime_error("Model has not been fitted.");
    }

    int numRows = testData.size();
    int numCols = testData[0].size();

    Eigen::MatrixXd X(numRows, numCols + 1); // Add a column for bias

    for (int i = 0; i < numRows; ++i) 
    {
        X(i, 0) = 1.0; // Bias term
        for (int j = 0; j < numCols; ++j) 
        {
            X(i, j + 1) = testData[i][j];
        }
    }

    Eigen::VectorXd predictions = X * m_coefficients;

    std::vector<double> result(numRows);
    for (int i = 0; i < numRows; ++i) 
    {
        result[i] = predictions(i);
    }

    return result;
}





/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
    try {
        // Check if the file path is empty
        if (filePath.empty()) {
            MessageBox::Show("Please browse and select the dataset file from your PC.");
            return {}; // Return an empty vector since there's no valid file path
        }

        // Attempt to open the file
        std::ifstream file(filePath);
        if (!file.is_open()) {
            MessageBox::Show("Failed to open the dataset file");
            return {}; // Return an empty vector since file couldn't be opened
        }
        // Load the dataset from the file path
        std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

        // Convert the dataset from strings to doubles
        std::vector<std::vector<double>> dataset;
        bool isFirstRow = true; // Flag to identify the first row

        for (const auto& row : data) {
            if (isFirstRow) {
                isFirstRow = false;
                continue; // Skip the first row (header)
            }

            std::vector<double> convertedRow;
            for (const auto& cell : row) {
                try {
                    double value = std::stod(cell);
                    convertedRow.push_back(value);
                }
                catch (const std::exception& e) {
                    // Handle the exception or set a default value
                    std::cerr << "Error converting value: " << cell << std::endl;
                    // You can choose to set a default value or handle the error as needed
                }
            }
            dataset.push_back(convertedRow);
        }

        // Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
        double trainRatio = trainingRatio * 0.01;

        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        // Fit the model to the training data
        fit(trainData, trainLabels);
        //fit(trainData, trainLabels, 0.1, 100);

        // Make predictions on the test data
        std::vector<double> testPredictions = predict(testData);
        //std::vector<double> testPredictions = predictGradient(testData);

        // Calculate evaluation metrics (e.g., MAE, MSE)
        double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
        double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
        double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

        // Make predictions on the training data
        std::vector<double> trainPredictions = predict(trainData);

        // Calculate evaluation metrics for training data
        double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
        double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
        double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

        MessageBox::Show("Run completed");
        return std::make_tuple(test_mae, test_rmse, test_rsquared,
            train_mae, train_rmse, train_rsquared,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        // Handle the exception
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            std::vector<double>(), std::vector<double>(),
            std::vector<double>(), std::vector<double>());
    }
}