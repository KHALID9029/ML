#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>

using namespace std;


									// EntropyFunctions class implementation //



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {
	int total_samples = y.size();
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	double entropy = 0.0;
	
	// Convert labels to unique integers and count their occurrences
	//TODO
	for (int i = 0; i < total_samples; i++) 
	{
		
			label_map[y[i]]++;
	}

	
	// Compute the probability and entropy
	//TODO
	for (const auto& label : label_map)
	{
		double probability = (double)label.second / (double)total_samples;
		entropy -= probability * log2(probability);
	}

	return entropy;
}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	int total_samples = idxs.size();
	double entropy = 0.0;

	// Convert labels to unique integers and count their occurrences
	//TODO
	for (int i = 0; i < total_samples; i++)
	{
	
			label_map[y[idxs[i]]]++;
		
	}


	// Compute the probability and entropy
	//TODO
	for (const auto& label : label_map)
	{
		double probability = (double)label.second / (double)total_samples;
		entropy -= probability * log2(probability);
	}


	return entropy;
}


