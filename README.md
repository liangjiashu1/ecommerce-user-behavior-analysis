# 📊 电商用户行为分析与购买预测项目

## 🔥 项目简介

本项目基于电商用户行为数据，分析影响用户购买的关键因素，并构建机器学习模型预测用户是否购买。

---

## 📁 项目结构

电商分析/
├── data/ 数据集
├── src/ 代码
├── images/ 可视化结果
└── README.md

---

## 📊 数据说明

包含8000条用户行为数据，主要字段：

- age：年龄
- gender：性别
- device_type：设备类型
- pages_viewed：浏览页数
- cart_items：加购数量
- bounce_rate：跳出率
- purchase：是否购买（目标变量）

---

## 📈 主要分析结果

- 加购行为是最重要的购买因素
- 跳出率越高，购买概率越低
- 浏览页数与购买正相关
- 年龄影响较弱

---

## 🤖 模型

- 模型：Logistic Regression（二分类）
- 任务：预测用户是否购买

---

## 📊 模型评估

- Accuracy
- Precision
- Recall
- F1-score
- ROC Curve

---

## 🧠 结论

用户行为特征比基础信息更能预测购买行为，其中“加购行为”是最关键变量。

---

## 🛠 技术栈

Python / Pandas / Sklearn / Matplotlib / Seaborn
