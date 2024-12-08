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
 

## **Trying the Project**
1. **Clone the Repository**:  
   ```bash
   git clone

    ```
2. **Install the Required Libraries**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the NoteBooks**:
    - [CART Implementation](ImplCART.ipynb)
    - [CatBoost Implementation](ImplCatBoost.ipynb)
    - [SVM Implementation](ImplSVM.ipynb)