
# 🧠 Parkinson's Disease Detection Using Machine Learning

## 📋 Overview

Parkinson's Disease (PD) is a progressive neurodegenerative disorder that affects movement, with early detection being crucial for effective management. This research project leverages advanced machine learning techniques to detect Parkinson's Disease using vocal biomarkers, demonstrating the potential of artificial intelligence in medical diagnostics.

## ✨ Project Features

### 🔍 Data Source
- **Dataset**: Parkinson's Disease Classification from the UCI Machine Learning Repository
- **Data Type**: Vocal feature measurements
- **Objective**: Binary classification (Parkinson's Disease: Yes/No)

### 🤖 Machine Learning Algorithms
1. **CART (Classification and Regression Tree)**
   - Decision Tree-based classification
   - Interpretable model with clear decision boundaries

2. **Support Vector Machines (SVM)**
   - Kernel-based classification technique
   - Effective for non-linear decision boundaries
   - Multiple kernel options (linear, polynomial, radial basis function)

3. **CatBoost**
   - Advanced gradient boosting algorithm
   - Handles categorical features efficiently
   - Robust to overfitting

### 🧬 Feature Processing Techniques

1. **All Features Approach**
   - Utilizes entire feature set without modification
   - Baseline performance evaluation

2. **Advanced Feature Selection**
   Techniques used to identify most predictive features:
   - Wrapper Method: Backward Elimination
   - Embedding Method: LassoCV
   - Statistical Method: ANOVA (Analysis of Variance)

3. **Dimensionality Reduction**
   - Principal Component Analysis (PCA)
   - Reduces feature space while preserving critical information
   - Mitigates curse of dimensionality

### 💻 Implementation Strategies
- Leveraging standard machine learning libraries
- Custom algorithm implementations
- Comprehensive performance comparison

## 🚀 Project Setup

### Prerequisites
- Python 3.8+
- Git
- Basic machine learning knowledge

### Installation Steps

1. **Repository Cloning**:
   ```bash
   git clone https://github.com/SecurDrgorP/ParkinsonML.git
   cd Parkinsons-Disease-Detection
   ```

2. **Virtual Environment** (Recommended):
   ```bash
   python -m venv env
   
   # Activate
   # Windows: env\Scripts\activate
   # macOS/Linux: source env/bin/activate
   ```

3. **Dependencies Installation**:
   ```bash
   pip install -r requirements.txt
   ```

### 🔬 Notebook Overview

| Notebook | Algorithm | Key Focus |
|----------|-----------|-----------|
| ImplCART.ipynb | Decision Tree | Interpretable Classification |
| ImplSVM.ipynb | Support Vector Machine | Complex Decision Boundaries |
| ImplCatBoost.ipynb | Gradient Boosting | Ensemble Learning |

### 🖥️ Running Notebooks
```bash
# Install Jupyter
pip install jupyter

# Launch
jupyter notebook
```

## 📊 Key Metrics Tracked
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Curve

## 🔮 Potential Impact
- Early Parkinson's Disease detection
- Assistance in clinical decision-making
- Demonstrating machine learning's potential in medical diagnostics

## 🤝 Contributions
Contributions, issues, and feature requests are welcome! Please check the issues page.


---

**Note**: This project is for research and educational purposes. Always consult healthcare professionals for medical advice.