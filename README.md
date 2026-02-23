# Credit Card Approval Prediction

##  Project Overview

This project builds a machine learning classification model to predict whether a credit card application will be approved or rejected.

Using applicant demographic and financial features, a Logistic Regression model was trained and optimized using cross-validation and hyperparameter tuning.

---

##  Problem Statement

Financial institutions must assess credit risk before approving credit card applications. This project simulates that decision-making process using supervised machine learning.

Target Variable:
- 1 → Approved
- 0 → Rejected

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- StandardScaler
- GridSearchCV

---

##  Project Workflow

1. Data Loading  
2. Missing Value Handling  
   - Categorical features → Mode imputation  
   - Numerical features → Mean imputation  
3. One-Hot Encoding of categorical variables  
4. Train-Test Split (67% train / 33% test)  
5. Feature Scaling using StandardScaler  
6. Logistic Regression Training  
7. Hyperparameter Tuning (tol, max_iter)  
8. Model Evaluation  

---

##  Hyperparameter Tuning

GridSearchCV was used with 5-fold cross-validation.

Parameter Grid:
- tol: [0.01, 0.001, 0.0001]
- max_iter: [100, 150, 200]

Best Parameters Found:
- max_iter = 100  
- tol = 0.01  

Best Cross-Validation Accuracy:
- 81.82%

---

##  Model Performance

Test Accuracy:
- **79.39%**

Confusion Matrix:

```
[[203   1]
 [  1 257]]
```

### Interpretation

- True Negatives: 203  
- False Positives: 1  
- False Negatives: 1  
- True Positives: 257  

The model misclassified only 2 applications in the test set, demonstrating strong predictive performance.

---

##  How To Run

```bash
pip install -r requirements.txt
python credit_card_model.py
```

---

##  Key Learning Outcomes

- Handling missing data in real-world datasets  
- Encoding categorical variables for ML models  
- Feature scaling for optimization stability  
- Logistic Regression for binary classification  
- Hyperparameter tuning using GridSearchCV  
- Cross-validation for model generalization  
- Confusion matrix interpretation  

---
