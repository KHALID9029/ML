# MachineLearningLab

A Windows desktop application for interactive machine learning experimentation, visualization, and evaluation. Built with C++/CLI and Windows Forms, it provides a user-friendly interface for running regression, classification, and clustering algorithms on custom datasets.

## Features

- **Regression Algorithms**
  - Linear Regression
  - Decision Tree Regression
  - KNN Regression

- **Classification Algorithms**
  - KNN Classifier
  - Decision Tree Classification
  - Logistic Regression

- **Clustering Algorithms**
  - K-Means
  - Fuzzy C-Means

- **Data Preprocessing**
  - Dataset loading from CSV
  - Normalization and scaling
  - Train/test split

- **Evaluation Metrics**
  - Regression: MAE, RMSE, R²
  - Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
  - Clustering: Davies-Bouldin Index, Silhouette Score

- **Visualization**
  - Parity plots for regression
  - Confusion matrices for classification
  - PCA-based cluster visualization

## Getting Started

### Prerequisites

- Windows OS
- Visual Studio (with C++/CLI support)
- .NET Framework

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/KHALID9029/ML.git
   ```
2. Open the solution in Visual Studio.
3. Build the project.

### Usage

1. Launch the application.
2. Use the tabs to select Regression, Classification, or Clustering.
3. Browse and load your dataset (CSV format).
4. Select the desired algorithm and configure parameters.
5. Run the algorithm to view results and visualizations.

## Project Structure

- `MachineLearningLab/src/`
  - `Regression/` - Regression algorithms
  - `Classification/` - Classification algorithms
  - `Clustering/` - Clustering algorithms
  - `Evaluation/` - Metrics for model evaluation
  - `DataUtils/` - Data loading and preprocessing
  - `Utils/` - Utility functions (e.g., PCA, similarity)
  - `MainForm.h/cpp` - Main application UI and logic

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Author

Khalid9029
