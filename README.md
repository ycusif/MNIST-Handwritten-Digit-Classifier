# 🚀 MNIST Handwritten Digit Classification

This project implements and compares two machine learning algorithms—**Logistic Regression** and **Support Vector Machine (SVM)**—to classify handwritten digits from the MNIST dataset.

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each sized 28×28 pixels.

---

## 📂 Project Structure

The project is organized for clarity and easy navigation:

MNIST-ML-Models/
│
├─ ml_project11.py        # Main Python script for data processing, training, and evaluation.
├─ results/               # Directory for generated output plots.
│  ├─ confusion_matrix.png # Plot of the Logistic Regression confusion matrix.
│  └─ sample_predictions.png # Plot of visual predictions on test samples.
└─ Report
markdown
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. DATA LOADING AND PREPROCESSING
print("--- 1. Data Loading and Preprocessing ---")
# Load the MNIST dataset (70,000 samples)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

x = mnist.data # Features (pixel values)
y = mnist.target.astype(int) # Labels (0-9)

# Normalize pixel values to the range [0, 1]
x = x / 255.0

# Split the data: 80% for training (56,000) and 20% for testing (14,000)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# --- 2. LOGISTIC REGRESSION MODEL (Trained on Full Data) ---
print("\n--- 2. Logistic Regression Model Training ---")

# Initialize and train Logistic Regression Classifier
log_clf = LogisticRegression(
    solver='lbfgs',
    max_iter=200, # Max iterations for solver convergence
    n_jobs=-1     # Use all available CPU cores
)

log_clf.fit(x_train, y_train)
log_pred = log_clf.predict(x_test)
log_accuracy = accuracy_score(y_test, log_pred)

# --- 3. SUPPORT VECTOR MACHINE (RBF) MODEL (Trained on Subset) ---
print("\n--- 3. Support Vector Machine (RBF) Model Training ---")

# Standardize the data (crucial for SVM)
scaler = StandardScaler()

# Use smaller subset (10,000 samples) for SVM due to high computational cost
X_train_small = x_train[:10000]
y_train_small = y_train[:10000]
x_train_scaled_small = scaler.fit_transform(X_train_small)

# Scale the full test set
x_test_scaled = scaler.transform(x_test)

# Train SVM with Radial Basis Function (RBF) kernel
svm_clf = SVC(kernel='rbf', gamma='scale')
svm_clf.fit(x_train_scaled_small, y_train_small)

# Predict
svm_pred = svm_clf.predict(x_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)

# --- 4. RESULTS AND EVALUATION ---
print("\n--- 4. Results and Evaluation ---")
print("Logistic Regression Accuracy:", log_accuracy)
print("SVM Accuracy:", svm_accuracy)

# Classification Report for Logistic Regression
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, log_pred))

# Confusion Matrix for Logistic Regression (Plot)
cm = confusion_matrix(y_test, log_pred)
# Code to plot the confusion matrix...

# Visualize 5 test samples (Plot)
# Code to plot the sample predictions...

### 📝 Workflow

1. **📥 Data Loading:** Fetches the MNIST dataset using `fetch_openml`.
2. **⚙️ Preprocessing:** Scales pixel values to `[0, 1]`. For SVM, further standardizes data with `StandardScaler`.
3. **🤖 Model Training:**
   - Logistic Regression: trained on the full training set.
   - SVM (RBF kernel): trained on a 10,000-sample subset for efficiency.
4. **📊 Evaluation:** Accuracy scores, classification report, confusion matrix, and sample test predictions.

---

## 🛠️ Requirements

Install required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
🏁 Getting Started
Clone the repository:

bash
Copy code
git clone https://github.com/<your-username>/MNIST-ML-Models.git
cd MNIST-ML-Models
Run the main script:

bash
Copy code
python ml_project11.py
The script will:

📥 Load and preprocess the MNIST data.

📊 Split the data into training (80%) and testing (20%) sets.

🤖 Train the Logistic Regression model and print its accuracy.

🤖 Train the SVM model (subset) and print its accuracy.

🖼️ Display a confusion matrix for Logistic Regression predictions.

🖼️ Visualize 5 sample test images with predicted vs. true labels.

📊 Results Summary
Model	Training Subset Size	Test Set Accuracy
Logistic Regression	56,000	~92.2%
Support Vector Machine	10,000	~96.6%

💡 The SVM achieves higher accuracy even with fewer training samples, demonstrating the power of non-linear models like RBF-kernel SVM for image classification.

🖼️ Visualizations
Confusion Matrix (Logistic Regression)
Shows correct predictions along the diagonal and misclassifications off-diagonal.

Sample Predictions
Displays 5 test images with predicted vs. true labels. ✅ Green = correct, ❌ Red = incorrect.

📜 License
This project is released under the MIT License.
