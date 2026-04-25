# 🚢 Titanic Survival Classification

## Problem Statement
Predict whether a passenger survived the Titanic disaster 
based on demographic and ticket information. A classic 
binary classification problem with real historical data.

## Dataset
- **Source:** Kaggle — Titanic: Machine Learning from Disaster
- **Size:** 891 rows, 12 features
- **Target:** Survived (0 = No, 1 = Yes)

## Approach
- Dropped Cabin (77% missing) and non-predictive columns 
  (Name, Ticket, PassengerId)
- Median imputation for Age, dropped 2 Embarked nulls (0.22%)
- Explicit label encoding for Sex, one-hot encoding for Embarked
- Feature scaling using StandardScaler
- Evaluated using classification report + confusion matrix

## Results
| Model | Accuracy | 
|---|---|
| Logistic Regression | 82.2% |
| **Random Forest** | **84.0%** |

**Random Forest Confusion Matrix:**
- True Negatives: 96 | False Positives: 9
- False Negatives: 15 | True Positives: 51

## Key Findings
- Random Forest outperformed Logistic Regression suggesting 
  non-linear relationships in survival patterns
- Sex and Pclass were the strongest survival predictors — 
  women and 1st class passengers had significantly higher 
  survival rates
- Precision and recall analysis confirmed the model is more 
  reliable at predicting non-survivors than survivors

## Libraries Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
