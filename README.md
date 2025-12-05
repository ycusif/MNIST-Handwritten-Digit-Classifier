# 🚀 MNIST Handwritten Digit Classification

This project implements and compares two machine learning algorithms—**Logistic Regression** and **Support Vector Machine (SVM)**—to classify handwritten digits from the MNIST dataset.

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each sized 28×28 pixels.

---

## 📂 Project Structure

The project is organized for clarity and easy navigation:

MNIST-ML-Models/
ml_project11.py       

results/               
   confusion_matrix.png 
   sample_predictions.png 

 Report


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
