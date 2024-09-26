#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
using namespace std;
using namespace System::Windows::Forms; // For MessageBox



///  DecisionTreeRegression class implementation  ///


// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{
}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {

	std::vector<double> predictions;
	
	// Implement the function
	// TODO

	for (int i = 0; i < X.size(); i++)
	{
		predictions.push_back(traverseTree(X[i], root));
	}
	return predictions;
}


// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {


	int split_idx = -1;
	double split_thresh = 0.0;

	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- Find the best split threshold for the current feature.
		--- grow the children that result from the split
	*/
	
	// TODO

	if (y.size() < min_samples_split || depth == max_depth)
	{
		return new Node(-1, -1, nullptr, nullptr, mean(y));
	}

	int n_samples = X.size();
	int n_features = X[0].size();

	double best_mse = 1e12;
	int best_split_idx = -1;
	double best_split_thresh = 0.0;

	for (int i = 0; i < n_features; i++)
	{
		vector<double> feat_values;
		for (int j = 0; j < n_samples; j++)
		{
			feat_values.push_back(X[j][i]);
		}

		set<double> unique_values(feat_values.begin(), feat_values.end());

		for (const auto& value : unique_values)
		{
			double mse = meanSquaredError(y, feat_values, value);
			if (mse < best_mse)
			{
				best_mse = mse;
				best_split_idx = i;
				best_split_thresh = value;
			}
		}
	}

	if (best_split_idx==-1)
	{
		return new Node(-1, -1, nullptr, nullptr, mean(y));
	}

	vector<vector<double>> left_X, right_X;
	vector<double> left_y, right_y;

	for (int i = 0; i < n_samples; i++)
	{
		if (X[i][best_split_idx] <= best_split_thresh)
		{
			left_X.push_back(X[i]);
			left_y.push_back(y[i]);
		}
		else
		{
			right_X.push_back(X[i]);
			right_y.push_back(y[i]);
		}
	}

	Node* left = growTree(left_X, left_y, depth+1);
	Node* right = growTree(right_X, right_y, depth+1);

	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {

	double mse = 0.0;
	
	// Calculate the mse
	// TODO

	vector<double> lefy_y, right_y;

	for (int i = 0; i < X_column.size(); i++)
	{
		if (X_column[i] <= split_thresh)
		{
			lefy_y.push_back(y[i]);
		}
		else
		{
			right_y.push_back(y[i]);
		}
	}	

	double lefy_mean = mean(lefy_y);
	double right_mean = mean(right_y);

	double left_mse = 0.0, right_mse = 0.0;

	for (int i = 0; i < lefy_y.size(); i++)
	{
		left_mse += pow(lefy_y[i] - lefy_mean, 2);
	}

	for (int i = 0; i < right_y.size(); i++)
	{
		right_mse += pow(right_y[i] - right_mean, 2);
	}	


	mse = (lefy_y.size() * left_mse + right_y.size() * right_mse) / (y.size());

	
	
	return mse;
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double meanValue = 0.0;
	
	// calculate the mean
	// TODO

	if (values.empty())
	{
		return 0.0;
	}

	//double sum = accumulate(values.begin(), values.end(), 0.0);

	double sum = 0.0;
	for (int i = 0; i < values.size(); i++)
	{
		sum += values[i];
	}

	meanValue = sum / values.size();
	
	return meanValue;
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {

	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/
	// TODO

	if (node->isLeafNode())
	{
		return node->value;
	}

	if (x[node->feature] <= node->threshold)
	{
		return traverseTree(x, node->left);
	}
	else
	{
		return traverseTree(x, node->right);
	}
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
	std::vector<double>, std::vector<double>,
	std::vector<double>, std::vector<double>>
	DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
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

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

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

