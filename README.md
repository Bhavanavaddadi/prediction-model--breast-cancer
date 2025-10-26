# 🩺 Breast Cancer Diagnosis Prediction using Machine Learning

A complete end-to-end **machine learning** solution for predicting whether a breast tumor is **malignant** or **benign** using the **Wisconsin Breast Cancer (Diagnostic)** dataset. The final model chosen is a **Support Vector Machine (SVM)**, with strong emphasis on minimizing **false negatives** (ensuring high recall for malignant cases).

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Technology Stack](#-technology-stack)
- [Workflow](#-workflow)
- [Model Evaluation](#-model-evaluation)
- [Usage](#️-usage)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🚀 Project Overview
The goal of this project is to build and evaluate a highly accurate machine learning model for **breast cancer classification**. In the medical domain, minimizing **False Negatives** (predicting benign when the tumor is malignant) is critical. Hence, the **Recall metric** for the malignant class is prioritized.

**Key workflow steps:**
- Data Cleaning & Exploration (EDA)
- Data Preprocessing (scaling)
- Model Training & Comparison (Logistic Regression, Random Forest, SVM)
- Model Selection (final: SVM)
- Model Persistence (save model & scaler using joblib)

---

## 📊 Dataset
**Dataset:** Wisconsin Breast Cancer (Diagnostic)

- **Source:** Available in `scikit-learn` or from public repositories.
- **Instances:** 569
- **Features:** 30 numerical features computed from digitized FNA images (e.g., mean radius, mean texture, mean smoothness).
- **Target Variable:** `diagnosis`
  - `0`: Malignant  
  - `1`: Benign

---

## 💻 Technology Stack
- Python 3.x
- Jupyter Notebook / Google Colab
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib (model persistence)

---

## ⚙️ Workflow
### **1. Data Loading**
Load the dataset into a Pandas DataFrame (e.g., via `sklearn.datasets.load_breast_cancer()`).

### **2. Exploratory Data Analysis (EDA)**
- Analyze class distribution (malignant vs benign)
- Visualize correlations (Seaborn heatmap)
- Inspect feature distributions (boxplots, histograms)

### **3. Data Preprocessing**
1. Separate features (`X`) and target (`y`).
2. Train-test split (80% train, 20% test, `random_state=42`).
3. Scale features using `StandardScaler` (fit on training data only).

### **4. Model Training & Comparison**
Train and compare the following models on the scaled training data:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### **5. Model Selection**
Models were evaluated using:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score

The **SVM** model achieved the best recall for the malignant class, making it the most reliable for diagnosis.

### **6. Model Persistence**
Save the trained model and scaler for reuse:

```python
from joblib import dump, load

# Save
 dump(svm_model, 'svm_model.joblib')
 dump(scaler, 'scaler.joblib')

# Load
 svm_model = load('svm_model.joblib')
 scaler = load('scaler.joblib')
```

---

## 📈 Model Evaluation
Models were evaluated on the test set. The key metric is **Recall for the Malignant class (Class 0)**.

> **Result:** The Support Vector Machine (SVM) outperformed Logistic Regression and Random Forest in both overall accuracy and malignant recall, ensuring the lowest risk of false negatives.
>
<img width="1790" height="387" alt="image" src="https://github.com/user-attachments/assets/0edd1b67-d9b6-4537-80bf-d4e1a816f8fd" />


---

## ▶️ Usage
### **1. Setup**
Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-folder>

# Optional: Create virtual environment
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# or (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

### **2. Run the Notebook**
Open and run `Breast_Cancer_Prediction.ipynb` in **Jupyter Notebook** or **Google Colab**. Follow the cells sequentially from data loading to prediction.

### **3. Make a Prediction**
Use the saved model to predict new observations:

```python
import numpy as np
from joblib import load

scaler = load('scaler.joblib')
model = load('svm_model.joblib')

# Example new observation (30 features)
new_obs = np.array([ ... ])  # replace with 30 feature values
new_obs_scaled = scaler.transform(new_obs.reshape(1, -1))
pred = model.predict(new_obs_scaled)[0]

print('Prediction:', 'Malignant' if pred == 0 else 'Benign')
```

---

## 🔮 Future Improvements
- **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` for optimal SVM parameters.
- **Feature Selection:** Apply Random Forest feature importances or Recursive Feature Elimination (RFE).
- **Web Interface:** Create a Streamlit or Flask app for real-time predictions.
- **Explainability:** Integrate SHAP or LIME for model interpretation.
- **Cross-Validation:** Use K-Fold CV for more robust evaluation.

---

## 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

**Created with ❤️ — You can copy this file as `README.md` for your GitHub repository.**
