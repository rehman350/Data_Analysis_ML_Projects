# E-commerce Customer Purchase Prediction

A machine learning project that predicts whether an online shopper will make a purchase or not, helping e-commerce businesses target the right customers at the right time.

---

## The Problem

Every e-commerce platform loses money on marketing because they show ads and discounts to everyone — including people who were never going to buy. This project solves that by predicting purchase intent before the customer leaves the site.

---

## Dataset

The dataset contains **12,330 real website session records** from an e-commerce platform. Each record represents one user visit and includes how many pages they browsed, how long they stayed, their bounce and exit rates, the monetary value of pages they visited, what month and type of visitor they were, and whether they actually made a purchase at the end.

The target variable was **Revenue** — True if the user purchased, False if they did not.

---

## Key Challenge — Class Imbalance

Only **15.5% of users actually purchased**. This meant a naive model could just predict "No Purchase" for everyone and still look 84% accurate — which is completely useless in practice. This was the core challenge of the project.

---

## What Was Observed in the Data

**PageValues was the strongest signal.** Users who purchased had significantly higher page values compared to those who did not. Non-buyers had a median PageValue of zero.

**November had the highest purchase rate at 25.4%**, while February was the lowest at just 1.6% — a clear seasonal pattern tied to year-end sales.

**New visitors converted more (24.9%) than returning visitors (13.9%).** This was unexpected but consistently shown in the data.

**BounceRates and ExitRates had a 0.91 correlation** — they were measuring almost the same thing, which was redundant for the model.

---

## What Was Done

**Preprocessing:** Categorical columns like Month and VisitorType were encoded into numbers. All features were scaled using StandardScaler. SMOTE was applied to fix the class imbalance by generating synthetic samples of the minority class, balancing training data to 8,338 samples per class.

**Feature Engineering:** Six new features were created from existing ones — total pages visited, total time spent, average time per page, a combined bounce-exit metric, a high-value user flag, and an engagement score. The engagement score and high-value user flag ended up in the top 3 most important features.

**Model Comparison:** Four models were trained and compared using 5-fold cross validation with F1 score as the metric, since accuracy was misleading on imbalanced data.

| Model | F1 Score |
|---|---|
| Logistic Regression | 0.8455 |
| Gradient Boosting | 0.9064 |
| XGBoost | 0.8870 |
| **Random Forest** | **0.9339 ✅** |

Random Forest was selected as the best model and further tuned using GridSearchCV across 108 parameter combinations.

---

## Final Results

| Metric | Score |
|---|---|
| Accuracy | 89% |
| F1 Score (Purchase) | 0.66 |
| F1 Score (No Purchase) | 0.93 |
| ROC-AUC | **0.918** |

The model correctly identified 271 out of 382 actual buyers in the test set. An AUC of 0.918 places this model in the production-ready range.

**Top features by importance:** PageValues, engagement score, is_high_value_user, Month, Administrative pages visited.

---

## Deployment

The trained model is deployed as an interactive web application using Streamlit. Users can input session behavior details and instantly get a purchase probability prediction with confidence level.
