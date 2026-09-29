# LendingClub Loan Default Prediction

Machine learning project analyzing historical LendingClub loan data to predict borrower default risk using statistical and gradient-boosting models.

## Overview

Built an end-to-end credit risk modeling pipeline using LendingClub loan data from 2007–2018. The analysis processes 176,083 completed loans and evaluates borrower and loan characteristics to identify factors associated with default risk.

The workflow includes:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering and one-hot encoding
- Feature standardization
- Class imbalance handling
- L2-regularized logistic regression
- LightGBM classification
- ROC-AUC evaluation and model comparison
- Feature importance analysis

## Results

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.7259 |
| LightGBM | 0.7133 |

The L2-regularized logistic regression achieved the strongest performance with a **0.7259 ROC-AUC**. Key predictors of loan default included FICO score, debt-to-income ratio, loan term, loan amount, and revolving credit characteristics.

## Technologies

**Python** · **Pandas** · **NumPy** · **scikit-learn** · **LightGBM** · **Matplotlib** · **Seaborn**

## Dataset

Historical LendingClub accepted-loan data from 2007–2018. The raw dataset contains more than 2 million loan records; the analysis uses a 200,000-record subset before filtering to completed loans.

The raw dataset is not included in this repository due to its size.

## Project Context

Developed as part of a five-person team for **PSTAT 100: Data Science Concepts and Analysis at UC Santa Barbara**.
