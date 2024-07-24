# Data Science Project: Effects of anxiety, depression and therapy on memory bias

## Overview
This project involves a series of practical examinations and model evaluations using various datasets to test out the impacts of anxiety, depression and therapy on negative memory bias. The tasks include correlation analysis, multiple regression, ANOVA, and classifier performance assessment. The project uses Python for all tasks and includes code to perform statistical analyses, regression modeling, and classifier evaluations.

## Dataset
Datasets for this project are located in the "data" folder and were provided by Creative Computing Institute at University of the Art London (2023).

## Preferred Language
Python

## Project Tasks

### 1. Correlation and Regression Analysis
- **Objective**: Analyze correlations and perform multiple regression on a provided dataset.
- **Steps**:
  - **Correlation Analysis**:
    - Perform a series of correlations on the dataset including Memory, Anxiety, Depression, and Self-Esteem.
  - **Multiple Regression**:
    - Examine the contribution of each independent variable to the prediction of Memory Bias.
    - Report how much of the variance is accounted for by the regression equation.
    - Predict Memory Bias for a person with specific scores for Anxiety, Depression, and Self-Esteem.
    - Determine the best predictor of Memory Bias.
    - Test if Anxiety is the salient predictor of Memory Bias by performing a stepwise regression.

### 2. Between-Subjects Design Analysis
- **Objective**: Analyze a 2x3 between-subjects design using ANOVA and simple effects analysis.
- **Dataset**: Improvement index based on Therapy (New vs. Old) and Duration (Short-term, Medium-term, Long-term).
- **Steps**:
  - **Descriptive Statistics**:
    - Generate a table of means and standard deviations for the dataset.
  - **ANOVA**:
    - Perform an ANOVA using the General Linear Model and report significant effects.
  - **Interaction Plotting**:
    - Plot interactions in two ways: Duration x Therapy and Therapy x Duration.
  - **Simple Effects Analysis**:
    - Analyze simple effects for Duration at New Therapy, Therapy at Medium-term, and Therapy at Long-term.
  - **Conclusions**:
    - Provide conclusions based on the simple effects analyses.

### 3. Classifier Evaluation
- **Objective**: Evaluate various classifiers on two selected datasets.
- **Steps**:
  - **Classifier Setup**:
    - Generate classifier objects with default hyperparameters:
      - LogisticRegression
      - LinearSVC
      - SVC
      - KNeighborsClassifier
      - Bayesian Logistic Regression
  - **Fitting**:
    - Fit each classifier on the selected datasets.
  - **Visualization**:
    - Provide plots to demonstrate the decision boundaries of each classifier.
  - **Performance Assessment**:
    - Comment on the performance of each classifier on the given dataset.

## Instructions for Running the Analysis

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Mclevesluna/Mobile-usage-and-search-performance.git
    cd your-repository-name
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Note: A GPU is not required for this project. It was developed on MacOS, but should work on other operating systems with minor adjustments.

3. **Run the Analysis**:
    Open the Jupyter notebook and run the cells to execute the analysis.

## Libraries Used
The following libraries are used in the project:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols

#Libraries for point 3
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
