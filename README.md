# 🛡️ DDoS Attack Classification (12-Class)

This project focuses on classifying **12 different types of Distributed Denial of Service (DDoS) attacks** using machine learning algorithms.
The workflow includes **data preprocessing, class balancing, feature scaling, model training, and evaluation**.

---

## 🔍 **Project Workflow**

### **1. Data Preprocessing**

To ensure clean and high-quality input data, the following preprocessing steps were applied:

#### ✅ **Remove Duplicate Rows**

* Eliminates identical records to avoid model bias.
* Ensures only unique entries are used for training.

#### ✅ **Remove Rows with Null / Missing Values**

* Rows containing `NaN` or missing values were dropped.
* Guarantees consistency and avoids model errors.

#### ✅ **Label Encoding**

* Converts categorical labels (attack types) into numeric form.
* Necessary for algorithms that only accept numerical inputs.

#### ✅ **Feature Scaling with StandardScaler**

* Standardizes features by removing mean and scaling to unit variance.
* Helps algorithms like Naive Bayes and XGBoost converge better.

#### ✅ **SMOTE (Synthetic Minority Oversampling Technique)**

* Balances the dataset by generating synthetic samples for minority attack classes.
* Removes class imbalance problems and improves model performance.

---

## 📊 **Dataset Split**

A **Stratified Train-Test Split** ensures equal class distribution in both sets.

* **Training Set:** 80%
* **Testing Set:** 20%
* Stratification maintains the same 12-class ratio across both sets.

---

## 🤖 **Machine Learning Models Implemented**

### **1. Naive Bayes**

* Fast, probabilistic classifier.
* Works well with large datasets.
* Good baseline for comparison.

### **2. Decision Tree**

* Creates a tree structure to classify attack types.
* Easy to interpret.
* Handles nonlinear relationships.

### **3. XGBoost**

* Boosted tree-based model with high accuracy.
* Handles imbalance well when combined with SMOTE.
* Typically provides the best performance.

---

## 🧪 **Model Evaluation**

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix (12-class visualization)
* Classification Report

---

## 📁 **Project Structure (Suggested)**

```
└── ddos-classification/
    ├── data/
    │   ├── raw_dataset.csv
    │   └── processed_dataset.csv
    ├── notebooks/
    │   ├── preprocessing.ipynb
    │   ├── modeling.ipynb
    │   └── evaluation.ipynb
    ├── models/
    │   ├── naive_bayes.pkl
    │   ├── decision_tree.pkl
    │   └── xgboost_model.pkl
    ├── scripts/
    │   ├── preprocess.py
    │   ├── train.py
    │   └── evaluate.py
    ├── README.md
    └── requirements.txt
```

---

## ⚙️ **Technologies Used**

* Python
* NumPy
* Pandas
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* XGBoost
* Matplotlib / Seaborn

---

## 🎯 **Goal**

The goal of this project is to build a reliable classifier capable of recognizing 12 different DDoS attack categories with strong accuracy—useful for modern cybersecurity defense systems.
