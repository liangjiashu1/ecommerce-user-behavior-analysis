import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)
# 1. 读取数据
df = pd.read_csv("../data/ecommerce_user_behavior_8000.csv")

# 2. 基本信息
print("数据形状:", df.shape)
print("列名:", df.columns)
print(df.head())

# 3. 购买分布
print(df['purchase'].value_counts())

# 4. 购买率
print("购买率：", df['purchase'].mean())

# 5. 购买 vs 未购买用户对比
print(df.groupby('purchase')[['age','pages_viewed','cart_items','bounce_rate']].mean())

# 6. 图：浏览页面 vs 购买
df.groupby('purchase')['pages_viewed'].mean().plot(kind='bar')
plt.title("Pages Viewed vs Purchase")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

corr = df[['age','pages_viewed','cart_items','bounce_rate','purchase']].corr()

print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

def segment(row):
    if row['purchase'] == 1 and row['cart_items'] > 3:
        return 'High Value'
    elif row['purchase'] == 1:
        return 'Normal'
    else:
        return 'Low Activity'

df['segment'] = df.apply(segment, axis=1)

print(df['segment'].value_counts())

df['segment'].value_counts().plot(kind='bar')
plt.title("User Segmentation")
plt.show()

print("\n===== 业务结论 =====")
print("1. 加购行为(cart_items)是影响购买的最强指标")
print("2. 跳出率(bounce_rate)越低，购买概率越高")
print("3. 浏览深度(pages_viewed)与购买呈正相关")
print("4. 年龄对购买影响较弱")

df['segment'].value_counts().plot(kind='bar')
plt.title("User Segmentation Distribution")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

features = [
    'age',
    'pages_viewed',
    'cart_items',
    'bounce_rate',
    'time_on_site',
    'avg_session_time',
    'previous_purchases'
]

df_model = df[features + ['purchase']].dropna()

X = df_model[features]
y = df_model['purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

import pandas as pd

importance = pd.DataFrame({
    'feature': features,
    'weight': model.coef_[0]
})

importance = importance.sort_values(by='weight', ascending=False)

print(importance)

import matplotlib.pyplot as plt

importance.plot(kind='bar', x='feature', y='weight', legend=False)
plt.title("Feature Importance for Purchase Prediction")
plt.show()

from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Heatmap")
plt.savefig(os.path.join(IMG_DIR, "heatmap.png"))
plt.close()

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"))
plt.close()

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(IMG_DIR, "roc_curve.png"))
plt.close()
