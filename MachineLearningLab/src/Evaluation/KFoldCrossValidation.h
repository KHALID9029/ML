#ifndef KFOLD_CROSS_VALIDATION_H
#define KFOLD_CROSS_VALIDATION_H

#include <vector>
#include "../Classification/DecisionTreeClassification.h"

class KFoldCrossValidation {
public:
    // Function to perform k-fold cross-validation
    static double performKFold(DecisionTreeClassification& model, std::vector<std::vector<double>>& X, std::vector<double>& y, int k);
};

#endif