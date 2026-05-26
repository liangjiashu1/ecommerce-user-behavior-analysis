# E-commerce User Behavior Analysis and Purchase Prediction  
（电商用户行为分析与购买预测）

A data analysis and machine learning project based on 8,000 e-commerce user behavior records.

基于 8000 条电商用户行为数据，分析影响购买转化的关键因素，并构建机器学习模型预测用户购买行为。

---

## Project Overview ｜ 项目简介

This project focuses on analyzing e-commerce user behavior patterns and predicting purchase probability.

项目主要目标：

- Analyze relationships between user behavior and purchase decisions  
  分析用户行为与购买决策之间的关系

- Identify key factors influencing conversion  
  识别影响转化率的关键因素

- Build a machine learning classification model  
  构建购买预测分类模型

- Extract actionable business insights  
  提供可落地业务建议

---

## Project Structure ｜ 项目结构

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

## Dataset Description ｜ 数据说明

The dataset contains 8,000 user behavior records.

主要字段：

- `age`：年龄
- `gender`：性别
- `device_type`：设备类型
- `time_on_site`：页面停留时间
- `pages_viewed`：浏览页数
- `cart_items`：购物车商品数量
- `previous_purchases`：历史购买次数
- `bounce_rate`：跳出率
- `avg_session_time`：平均会话时长

Target Variable：

- `purchase`（是否购买）

---

## Exploratory Data Analysis ｜ 数据探索分析

Key findings:

- Cart activity is the strongest purchase indicator  
  加购行为是最强购买信号

- Higher bounce rate reduces conversion probability  
  跳出率越高，购买概率越低

- More page views increase purchase likelihood  
  浏览深度越高，购买概率越高

- Demographic features have weaker predictive power  
  用户基础属性影响较弱

---

## Machine Learning Model ｜ 模型构建

Model Used:

- Logistic Regression

Workflow:

1. Data preprocessing（数据预处理）
2. Feature analysis（特征分析）
3. Train-test split（训练测试划分）
4. Model training（模型训练）
5. Evaluation（模型评估）

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

---

## Feature Importance ｜ 特征重要性

Most influential features:

- `cart_items`
- `previous_purchases`
- `time_on_site`
- `pages_viewed`
- `bounce_rate`

Behavioral features are significantly more predictive than demographic features.

用户行为特征明显比静态人口属性更具预测价值。

---

## Business Insights ｜ 业务结论

The analysis suggests:

- Cart activity is the strongest conversion signal  
- Lower bounce rate improves conversion performance  
- Higher engagement increases purchase probability  
- Behavioral data provides stronger business value

业务建议：

- 优化购物车触达策略
- 降低页面跳出率
- 提升用户浏览深度
- 强化用户行为驱动运营策略

---

## Technologies Used ｜ 技术栈

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Results ｜ 项目成果

Completed a full end-to-end workflow:

- Data cleaning
- Exploratory analysis
- Predictive modeling
- Visualization
- Business interpretation

实现完整数据分析闭环，具备实际项目展示价值。
