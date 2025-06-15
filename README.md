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
```

### 2. Load the Dataset
We use Scikit-learn’s built-in Iris dataset. It contains:
- 150 samples
- 4 features (sepal/petal length and width)
- 3 classes (Setosa, Versicolor, Virginica)

```python
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
```

### 3.Split the Dataset
Split data into 80% training and 20% testing.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Train the Model
Create and train the Decision Tree Classifier using entropy and max depth of 3.

```python
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)
```

### 5. Make Predictions
Use the trained model to predict values on the test set.

```python
y_pred = model.predict(X_test)
```

### 6. Evaluate the Model
Evaluate using accuracy, confusion matrix, and classification report.

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

### 7. Visualize the Decision Tree
Visualize the trained model.

```python
plt.figure(figsize=(15, 10))
plot_tree(model, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          filled=True, 
          rounded=True, 
          fontsize=12)
plt.title("Decision Tree Visualization - Iris Dataset")
plt.show()
```

## Results:
The model demonstrates strong performance, achieving high accuracy. The confusion matrix and classification report show minimal misclassification. Visualization of the decision tree provides clear insights into feature-based decision-making, especially the importance of petal width and length.

##  Conclusion:
This project successfully demonstrates the implementation of a Decision Tree Classifier using the Iris dataset. The model was trained, evaluated, and visualized using Scikit-learn. It achieved high accuracy and provided clear insights through metrics and a visual decision tree. 

## Decision Tree Representation:
### OUTPUT:

![Screenshot (213)](https://github.com/user-attachments/assets/e645636f-351b-4398-9887-995f38709647)
