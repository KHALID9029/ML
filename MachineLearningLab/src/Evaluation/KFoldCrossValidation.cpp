#include "KFoldCrossValidation.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "../Classification/DecisionTreeClassification.h"

using namespace std;

// Function to perform k-fold cross-validation
double KFoldCrossValidation::performKFold(DecisionTreeClassification& model, std::vector<std::vector<double>>& X, std::vector<double>& y, int k) {
	int num_samples = X.size();
	int num_samples_per_fold = num_samples / k;
	double accuracy = 0.0;

	// Shuffle the data
	vector<int> indices(num_samples);
	iota(indices.begin(), indices.end(), 0);
	random_device rd;
	mt19937 g(rd());
	shuffle(indices.begin(), indices.end(), g);

	// Perform k-fold cross-validation
	for (int i = 0; i < k; i++) {
		// Split the data into training and testing sets
		vector<vector<double>> X_train, X_test;
		vector<double> y_train, y_test;
		for (int j = 0; j < num_samples; j++) {
			if (j >= i * num_samples_per_fold && j < (i + 1) * num_samples_per_fold) {
				X_test.push_back(X[indices[j]]);
				y_test.push_back(y[indices[j]]);
			}
			else {
				X_train.push_back(X[indices[j]]);
				y_train.push_back(y[indices[j]]);
			}
		}

		// Fit the model on the training data
		model.fit(X_train, y_train);

		// Predict the labels for the test data
		vector<double> y_pred = model.predict(X_test);

		// Calculate the accuracy of the model
		double correct = 0;
		for (int j = 0; j < y_test.size(); j++) {
			if (y_test[j] == y_pred[j]) {
				correct++;
			}
		}
		accuracy += correct / y_test.size();
	}

	return accuracy / k;
}