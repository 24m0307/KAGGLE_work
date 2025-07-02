# 🧠 Personality Prediction: Extrovert vs Introvert using ML

This project explores and models personality traits (extroversion vs introversion) using machine learning techniques. The dataset used is sourced from Kaggle and contains features that help identify whether a person is an **extrovert** or an **introvert** based on survey responses.

Two models — **Logistic Regression** and **Random Forest Classifier** — are trained and evaluated for classification performance.

---

## 📁 Dataset

**Source**: [Kaggle - Extrovert vs Introvert Personality Dataset](https://www.kaggle.com/)  
**Target column**: `target` (binary: `0 = Introvert`, `1 = Extrovert`)  
**Features**: Multiple survey-based categorical and numerical features (e.g., habits, preferences, and behaviors)

---

## 🧪 Project Workflow

### 1. **Data Loading & Cleaning**
- Load dataset using Pandas
- Handle missing values and invalid entries
- Convert labels (`0` / `1`) to `no` / `yes` if needed for interpretability

### 2. **EDA (Exploratory Data Analysis)**
- Distribution plots of personality types
- Correlation heatmap and feature importance
- Class imbalance check

### 3. **Preprocessing Pipeline**
- One-hot encoding for categorical features (`OneHotEncoder`)
- Standard scaling for numerical features (`StandardScaler`)
- Combined into a unified pipeline using `ColumnTransformer`

### 4. **Train-Test Split**
- 80/20 train-test split using `train_test_split` from `sklearn`

### 5. **Model Training**
- **Logistic Regression**
- **Random Forest Classifier**

Both models are trained using the encoded and scaled features.

### 6. **Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Curve

---

## 🛠️ Requirements

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
