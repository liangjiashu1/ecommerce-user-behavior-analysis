# E-commerce User Behavior Analysis and Purchase Prediction

A data analysis and machine learning project based on 8,000 e-commerce user behavior records.  
The project explores user behavior patterns, identifies key factors influencing purchase decisions, and builds a classification model to predict purchase probability.

---

## Project Overview

This project focuses on analyzing e-commerce user behavior data to understand conversion-driving factors and build a predictive model for purchase behavior.

Main objectives:

- Analyze relationships between user behavior and purchase decisions
- Identify the most influential behavioral features
- Build a machine learning classification model
- Extract actionable business insights

---

## Project Structure

```text
ecommerce-analysis/
│
├── data/
│   └── ecommerce_user_behavior_8000.csv
│
├── src/
│   └── ecommerce_analysis.py
│
├── images/
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt
└── README.md
```

---

## Dataset Description

The dataset contains 8,000 user records with behavioral and demographic information.

Features include:

- `age`
- `gender`
- `device_type`
- `time_on_site`
- `pages_viewed`
- `cart_items`
- `previous_purchases`
- `bounce_rate`
- `avg_session_time`

Target variable:

- `purchase`

---

## Exploratory Data Analysis

Key findings:

- Cart activity shows the strongest positive relationship with purchase behavior
- Higher bounce rates significantly reduce purchase probability
- Page depth positively correlates with conversion
- Demographic features have relatively weak predictive power

---

## Machine Learning Model

Model used:

- Logistic Regression

Workflow:

1. Data preprocessing
2. Feature selection
3. Train-test split
4. Model training
5. Performance evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

---

## Feature Importance

Most influential variables:

- `cart_items`
- `previous_purchases`
- `time_on_site`
- `pages_viewed`
- `bounce_rate`

Behavioral signals are significantly more predictive than demographic features.

---

## Business Insights

The analysis suggests:

- Cart activity is the strongest purchase signal
- Reducing bounce rate can improve conversion performance
- Improving browsing engagement increases purchase likelihood
- User behavior data provides stronger predictive value than static profile attributes

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Results

The project successfully implements an end-to-end workflow including:

- Data cleaning
- Exploratory analysis
- Predictive modeling
- Visualization
- Business interpretation

---

## Author

Undergraduate Project  
Central University of Finance and Economics  
Information and Computing Science
