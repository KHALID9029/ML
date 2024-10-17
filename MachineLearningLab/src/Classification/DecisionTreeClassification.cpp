#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
using namespace std;
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	
	// Implement the function
	// TODO
	for (int i = 0; i < X.size(); i++)
	{
		predictions.push_back(traverseTree(X[i], root));
	}

	return predictions;
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	
	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- greedily select the best split according to information gain
		---grow the children that result from the split
	*/
	
	double best_gain = -1; // set the best gain to -1
	int split_idx = NULL; // split index
	double split_thresh = NULL; // split threshold
	
	// TODO
	if (y.size() < min_samples_split || depth == max_depth)
	{
		return new Node(-1,-1,nullptr,nullptr, mostCommonlLabel(y));
	}

	int n_features = X[0].size();
	int n_samples = X.size();

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
			double gain = informationGain(y, feat_values, value);
			if (gain > best_gain)
			{
				best_gain = gain;
				split_idx = i;
				split_thresh = value;
			}
		}
	}

	if (best_gain == -1)
	{
		return new Node(-1,-1,nullptr,nullptr, mostCommonlLabel(y));
	}

	vector<vector<double>> left_child_x, right_child_x;
	vector<double> left_child_y, right_child_y;

	for (int i = 0; i < n_samples; i++)
	{
		if (X[i][split_idx] <= split_thresh)
		{
			left_child_x.push_back(X[i]);
			left_child_y.push_back(y[i]);
		}
		else
		{
			right_child_x.push_back(X[i]);
			right_child_y.push_back(y[i]);
		}
	}
	
	Node* left = growTree(left_child_x, left_child_y, depth+1); // grow the left tree
	Node* right = growTree(right_child_x, right_child_y, depth+1);  // grow the right tree
	
	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	double parent_entropy = EntropyFunctions::entropy(y);

	/* Implement the following:
	   --- generate split
	   --- compute the weighted avg. of the loss for the children
	   --- information gain is difference in loss before vs. after split
	*/
	double ig = 0.0;
	
	// TODO
	vector<int> left_child, right_child;
	for (int i = 0; i < X_column.size(); i++) 
	{
		if (X_column[i] <= split_thresh) 
		{
			left_child.push_back(i);
		}
		else 
		{
			right_child.push_back(i);
		}
	}

	if (left_child.size() == 0 || right_child.size() == 0)
	{
		return 0;
	}

	double left_entropy = EntropyFunctions::entropy(y,left_child);
	double right_entropy = EntropyFunctions::entropy(y,right_child);

	double num_left = left_child.size();
	double num_right = right_child.size();
	double total = y.size();

	double avg_child_entropy = (num_left / total) * left_entropy + (num_right / total) * right_entropy;

	ig = parent_entropy - avg_child_entropy;

	return ig;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {	
	double most_common = 0.0;
	
	// TODO
	unordered_map<double, int> label_map;
	for (int i = 0; i < y.size(); i++)
	{
		
		label_map[y[i]]++;
		
	}

	int max_count = -1;
	for (const auto& label : label_map)
	{
		if (label.second > max_count)
		{
			most_common = label.first;
			max_count = label.second;
		}
	}
	return most_common;
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

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
	
	return 0.0;
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
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
		fit(trainData, trainLabels);//

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