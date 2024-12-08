# Parkinson's Disease Detection Using Machine Learning  

## **Overview**  
This project aims to detect Parkinson's Disease using vocal data by applying machine learning algorithms. Three models — CART (Decision Tree), SVM (Support Vector Machines), and CatBoost — are implemented across different feature processing methods. Both standard libraries and custom-built implementations are utilized to evaluate model performance.  

## **Project Features**  
- **Dataset**: Parkinson’s Disease Classification from the UCI Machine Learning Repository.  
- **Algorithms**:  
  - CART (Decision Tree)  
  - Support Vector Machines (SVM)  
  - CatBoost  
- **Methods**:  
  1. **All Features** (`Toutes caractéristiques`): Using all features in the dataset.  
  2. **Feature Selection** (`La Sélection des caractéristiques`): Combining features from:  
     - Wrapper Method (Backward Elimination)  
     - Embedding Method (LassoCV)  
     - ANOVA  
  3. **Dimensionality Reduction** (`Réduction des données`): Principal Component Analysis (PCA).  
- **Implementation Approaches**:  
  - Using standard libraries like scikit-learn and CatBoost.  
  - Using custom-built implementations for each algorithm.  
 


## **Getting Started with Parkinson's Disease Detection Project**

### Prerequisites
- Python 3.8+ installed
- Git installed on your system
- Basic understanding of Python and machine learning concepts

### 🚀 Project Setup

1. **Clone the Repository**:
   ```bash
   # Clone the project repository
   git clone https://github.com/AbdulMoizAli/Parkinsons-Disease-Detection.git
   
   # Move into the project directory
   cd Parkinsons-Disease-Detection
   ```

2. **Set Up a Virtual Environment** (Recommended):
   ```bash
   # Create a new virtual environment
   python -m venv env
   
   # Activate the virtual environment
   # On Windows
   env\Scripts\activate
   
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   # Install required Python libraries
   pip install -r requirements.txt
   ```

### 🔬 Exploring Model Implementations

This project includes three different machine learning model implementations for Parkinson's Disease detection:

| Notebook | Description | Algorithm |
|----------|-------------|-----------|
| [CART Implementation](ImplCART.ipynb) | Classification and Regression Tree | Decision Tree |
| [CatBoost Implementation](ImplCatBoost.ipynb) | Gradient Boosting Algorithm | Ensemble Learning |
| [SVM Implementation](ImplSVM.ipynb) | Support Vector Machine | Kernel-based Classification |

### 🖥️ Running the Notebooks

To run the notebooks, ensure you have Jupyter Notebook or Jupyter Lab installed:

```bash
# If not already installed
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook
```

### 💡 Additional Recommendations
- Ensure you have the latest version of pip: `python -m pip install --upgrade pip`
- Check the project's documentation for specific model usage and dataset details
- Consider using Jupyter Lab for a more integrated development experience

### 🛠️ Troubleshooting
- If you encounter library compatibility issues, verify your Python version
- Make sure all dependencies in `requirements.txt` are compatible
- Check the project's GitHub issues for known problems and solutions