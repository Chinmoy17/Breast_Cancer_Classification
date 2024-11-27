# Breast Cancer Classification Project

This project is a comprehensive implementation of various machine learning models to classify breast cancer using the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). The dataset contains 31 features and a target variable indicating whether the cancer is malignant (M) or benign (B).

---

## Table of Contents

- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Implementation](#implementation)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Machine Learning Models](#2-machine-learning-models)
  - [3. Model Evaluation](#3-model-evaluation)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Acknowledgments](#acknowledgments)

---

## Dataset Description

The dataset consists of 569 samples of breast cancer data, each with the following:
- **Features**: 30 numerical features derived from cell nuclei of breast mass images.
- **Target**: A binary classification label (`M` for malignant, `B` for benign).

---


## Requirements

To run the project, install the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib` (optional, for visualization)




Install all dependencies using:

bash
pip install -r requirements.txt


---

**Cell 6: Implementation**


## Implementation

### 1. Data Preprocessing

- **Label Encoding**: Transformed the target variable (`diagnosis`) from categorical to numerical.
- **Splitting**: Divided the dataset into training and testing sets (80-20 split).
- **Features and Target**: Used all 30 features to predict the binary target variable.

### 2. Machine Learning Models

The following models were implemented and compared:

1. **Random Forest Classifier**
2. **K-Nearest Neighbors (KNN)**
3. **Naive Bayes with Bagging**
4. **Decision Tree Classifier**
5. **Gradient Boosting Machine (XGBoost)**

### 3. Model Evaluation

- **Metrics Used**: 
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Results**: Evaluated on a test set of 114 samples.

## Results

| Model                        | Accuracy | Precision | Recall | F1-Score |
|------------------------------|----------|-----------|--------|----------|
| Random Forest Classifier     | 96.49%   | 97%       | 96%    | 96%      |
| K-Nearest Neighbors (KNN)    | 95.61%   | 96%       | 96%    | 96%      |
| Naive Bayes with Bagging     | 97.37%   | 97%       | 97%    | 97%      |
| Decision Tree Classifier     | 94.74%   | 95%       | 95%    | 95%      |
| Gradient Boosting (XGBoost)  | 95.61%   | 96%       | 95%    | 96%      |

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/Chinmoy17/breast-cancer-classification.git


---

**Cell 9: Acknowledgments**


## Acknowledgments

- **Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
- Inspired by real-world applications in cancer diagnosis and medical data analysis.

---

Feel free to fork this project and contribute!


