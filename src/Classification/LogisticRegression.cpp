#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map> 
#include <unordered_set>

using namespace std;

using namespace System::Windows::Forms; // For MessageBox


///  LogisticRegression class implementation  ///
// Constractor

LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {}

// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) 
{
    int num_samples = X_train.size();
    int num_features = X_train[0].size();
    int num_classes = set<double>(y_train.begin(), y_train.end()).size();

    weights.resize(num_classes, vector<double>(num_features + 1, 0.0)); // weights for each class +1 for the bias term

    for (int class_label = 0; class_label < num_classes; ++class_label) // each Label
    {   
        vector<double> binary_y_train(num_samples); // converting in binary classification
        for (int i = 0; i < num_samples; i++) 
        {
            binary_y_train[i] = (y_train[i] == class_label + 1) ? 1.0 : 0.0;
        }

        for (int epoch = 0; epoch < num_epochs; ++epoch) // gradient descent
        { 
            for (int i = 0; i < num_samples; ++i) // each training sample
            { 
                double weighted_sum = weights[class_label][0]; // bias term
                for (int j = 0; j < num_features; ++j) 
                {
                    weighted_sum += weights[class_label][j + 1] * X_train[i][j];
                }

                double predicted = sigmoid(weighted_sum);


                double error = predicted - binary_y_train[i];

                weights[class_label][0] -= learning_rate * error;


                for (int j = 0; j < num_features; ++j) 
                {
                    weights[class_label][j + 1] -= learning_rate * error * X_train[i][j];
                }
            }
        }
    }
}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) 
{
    vector<double> predictions;
    int num_samples = X_test.size();
    int num_features = X_test[0].size();
    int num_classes = weights.size();

    for (int i = 0; i < num_samples; ++i) // each test sample
    { 
        vector<double> class_scores(num_classes, 0.0);

        for (int class_label = 0; class_label < num_classes; ++class_label) 
        {
            double weighted_sum = weights[class_label][0]; // bias term
            for (int j = 0; j < num_features; ++j) 
            {
                weighted_sum += weights[class_label][j + 1] * X_test[i][j];
            }
            class_scores[class_label] = weighted_sum; // score for this class
        }

        predictions.push_back(distance(class_scores.begin(), max_element(class_scores.begin(), class_scores.end())) + 1); // predict the class label with the highest score
    }

    return predictions;
    /*Predicting for each test sample : For each test sample, the model calculates a score for each class
    using the weighted sum of the input features(and bias term).
    Class with the highest score : The class with the highest score is selected as the predicted label for that sample.
    Appending the predicted class : The predicted class is then added to the predictions vector, which is returned at the end.*/
}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

    DataPreprocessor DataPreprocessor;
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

        std::vector<std::vector<double>> dataset; // Create an empty dataset vector
        DataLoader::loadAndPreprocessDataset(filePath, dataset);

        // Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
        double trainRatio = trainingRatio * 0.01;

        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        // Fit the model to the training data
        fit(trainData, trainLabels);

        // Make predictions on the test data
        std::vector<double> testPredictions = predict(testData);

        // Calculate accuracy using the true labels and predicted labels for the test data
        double test_accuracy = Metrics::accuracy(testLabels, testPredictions);

        // Make predictions on the training data
        std::vector<double> trainPredictions = predict(trainData);

        // Calculate accuracy using the true labels and predicted labels for the training data
        double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

        MessageBox::Show("Run completed");
        return std::make_tuple(train_accuracy, test_accuracy,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        // Handle the exception
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, std::vector<double>(),
            std::vector<double>(), std::vector<double>(),
            std::vector<double>());
    }
}