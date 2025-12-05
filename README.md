# 🚀 MNIST Handwritten Digit Classification

This project implements and compares two classic machine learning algorithms—**Logistic Regression** and **Support Vector Machine (SVM)**—to classify handwritten digits from the widely-used **MNIST** dataset.

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each sized $28 \times 28$ pixels.

-----

## 📂 Project Structure

The repository is organized for clarity and easy navigation:

```
MNIST-ML-Models/
│ 
├─ ml_project11.py        # Main Python script 
├─ requirements.txt       # Python dependencies 
├─ results/               # Generated plots (confusion matrix, sample predictions) 
└─ notebooks/             # Optional exploratory notebooks (currently empty)
```

-----

## 📝 Workflow

The `ml_project11.py` script executes a complete machine learning pipeline:

1.  **📥 Data Loading:** Fetch the MNIST dataset using `fetch_openml`.
2.  **⚙️ Preprocessing:**
      * Scale pixel values to the range $[0, 1]$.
      * Split the dataset 80% for training and 20% for testing.
      * Standardize features using `StandardScaler` specifically for the SVM model.
3.  **🤖 Model Training:**
      * **Logistic Regression:** Trained on the full 56,000-sample training set.
      * **SVM (RBF kernel):** Trained on a smaller 10,000-sample subset for efficiency.
4.  **📊 Evaluation:** Calculate accuracy scores for both models, generate a confusion matrix, and visualize sample predictions.

-----

## 🛠️ Requirements

The required Python dependencies for this project are:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

-----

## 🏁 Getting Started

### Clone the repository:

```bash
git clone https://github.com/<your-username>/MNIST-ML-Models.git
cd MNIST-ML-Models
```

### Run the main script:

```bash
python ml_project11.py
```

The script will:

  * 📥 Load and preprocess the MNIST data.
  * 🤖 Train the Logistic Regression model and print its accuracy.
  * 🤖 Train the SVM model (subset) and print its accuracy.
  * 🖼️ Display a confusion matrix for Logistic Regression.
  * 🖼️ Visualize 5 test images with predicted vs. true labels.

-----

## 📊 Results Summary

| Model | Training Subset Size | Test Set Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** | 56,000 | \~92.2% |
| **Support Vector Machine (RBF)** | 10,000 | \~96.6% |

💡 SVM achieves higher accuracy (approximately 96.6%) even with significantly fewer samples than the Logistic Regression model (approximately 92.2%), demonstrating the power of non-linear RBF-kernel SVM for image classification tasks.

-----

## 🖼️ Visualizations

The script generates plots to assess model performance visually.

### Confusion Matrix (Logistic Regression)

A heatmap showing correct predictions along the diagonal and highlighting misclassifications off-diagonal.

### Sample Predictions

Displays the first 5 test images with the model's predicted label versus the true label for visual inspection of the model's performance.

-----

## 📜 License

This project is licensed under the **MIT License**.
