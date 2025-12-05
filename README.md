# 🚀 MNIST Handwritten Digit Classification

This project implements and compares two classic machine learning algorithms—**Logistic Regression** and **Support Vector Machine (SVM)**—to classify handwritten digits from the widely-used **MNIST** dataset.

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each sized $28 \times 28$ pixels.

---

## 📂 Project Structure

The repository is organized as follows:

MNIST-ML-Models/│├─ ml_project11.py        # Main Python script for data processing, training, and evaluation.├─ results/               # Directory for generated output plots.│  ├─ confusion_matrix.png # Heatmap of the Logistic Regression confusion matrix.│  └─ sample_predictions.png # Plot of visual predictions on test samples.└─ README.md              # Project overview and usage guide (this file).
---

## 🛠️ Setup and Installation

### Prerequisites

This project requires Python and the following libraries, which can be installed via pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
ExecutionClone the repository:Bashgit clone [YOUR_REPOSITORY_LINK]
cd MNIST-ML-Models
Run the main script:Bashpython ml_project11.py
The script will print the model accuracies to the console and automatically save the evaluation plots into the results/ directory.💻 Code Breakdown (ml_project11.py)ImportsThe script uses standard scientific and machine learning libraries:Pythonimport pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
1. Data Loading and PreprocessingThe MNIST data is fetched and preprocessed:Normalization: Pixel values are scaled to the range $[0, 1]$ (x = x / 255.0).Splitting: The data is split 80% for training and 20% for testing (test_size=0.2).Feature Scaling: A StandardScaler is applied to the training and testing data for the SVM model, as it is sensitive to feature scaling.2. Model Training and ComparisonLogistic Regression (Baseline Model)A linear classifier trained on the full training set:Pythonlog_clf = LogisticRegression(solver='lbfgs', max_iter=200, n_jobs=-1)
log_clf.fit(x_train, y_train)
log_pred = log_clf.predict(x_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
Support Vector Machine (RBF Kernel)A powerful non-linear classifier trained on a smaller subset of 10,000 samples due to its high computational cost:PythonX_train_small = x_train[:10000]
y_train_small = y_train[:10000]
# ... scaling code ...
svm_clf = SVC(kernel='rbf', gamma='scale')
svm_clf.fit(x_train_scaled_small, y_train_small)
svm_pred = svm_clf.predict(x_test_scaled)
3. Evaluation and VisualizationThe script generates a confusion matrix heatmap for the Logistic Regression model and plots the first 5 predictions from the test set for visual verification:Pythoncm = confusion_matrix(y_test, log_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig('results/confusion_matrix.png') 
plt.show()

# Sample Predictions
for i in range(5):
    # ... plot image ...
    plt.title(f"Pred: {log_pred[i]}, True: {y_test[i]}")
    plt.savefig(f'results/sample_prediction_{i}.png')
    plt.show()
📊 Expected ResultsModelTraining Data SizeTypical Test AccuracyLogistic Regression56,000 samples (Full)~ 92.2%SVM (RBF Kernel)10,000 samples (Subset)~ 96.0%
