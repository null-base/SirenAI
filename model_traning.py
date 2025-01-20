import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# データをロード
data = pd.read_csv("features.csv")
X = data.iloc[:, :-1]  # 特徴量
y = data['label']  # ラベル

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレスト分類器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# テスト
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# モデルを保存
with open("siren_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to siren_model.pkl")
