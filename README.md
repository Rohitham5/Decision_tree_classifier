# Decision_tree_classifier

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MADDINENI ROHITHA

*INTERN ID*: CT06DL736

*DOMAIN*: MACHINE LEARNING

*DURATION*: 6 WEEEKS

*MENTOR*: NEELA SANTOSH

# Decision Tree Classification using Scikit-learn

This project demonstrates how to implement and visualize a Decision Tree Classifier using Python's Scikit-learn library. It is developed as part of the internship deliverables for CodTech and fulfills the requirement of building and visualizing a Decision Tree model with analysis on a chosen dataset.

## Project Objective

The primary objective of this project is to:
- Train a Decision Tree model to classify data
- Visualize the structure of the tree
- Analyze its performance using various evaluation metrics

We use the well-known Iris dataset for classification, which contains 150 samples of iris flowers, with 4 features per sample and 3 distinct species (Setosa, Versicolor, Virginica).

##  Tools & Technologies Used

- **Scikit-learn** – for model building and evaluation
- **Pandas** – for data manipulation
- **NumPy** – for numerical operations
- **Matplotlib** – for plotting the decision tree
- **Jupyter Notebook** – for development and visualization

## Code Breakdown

### 1. Import Required Libraries
We begin by importing essential libraries for data manipulation, model training, evaluation, and visualization.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

