# ==============================
# 📌 IMPORT LIBRARIES
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==============================
# 📌 LOAD DATA
# ==============================
data = pd.read_csv('energy_data.csv')

# ==============================
# 📌 DATA CLEANING + FEATURES
# ==============================
data = data.replace('?', pd.NA)

# Convert date
data['date'] = pd.to_datetime(data['date'])
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month

# Convert time → hour
data['hour'] = pd.to_datetime(data['time'], format='%H:%M').dt.hour

# Drop original columns
data = data.drop(['date', 'time'], axis=1)

# Convert numeric safely
data = data.apply(lambda col: pd.to_numeric(col, errors='coerce'))

# Fill missing
data = data.fillna(data.mean())


# ==============================
# 📌 SPLIT DATA
# ==============================
X = data.drop('energy', axis=1)
y = data['energy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 📌 TRAIN MODELS
# ==============================
lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)


# ==============================
# 📌 EVALUATION
# ==============================
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

print("\n==============================")
print("📊 MODEL PERFORMANCE")
print("==============================")

print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("Linear Regression R2:", r2_score(y_test, lr_pred))

print("\nRandom Forest MAE:", mean_absolute_error(y_test, rf_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))


# ==============================
# 📌 GRAPH
# ==============================
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(lr_pred, label="Linear", marker='x')
plt.plot(rf_pred, label="RandomForest", marker='s')

plt.title("Energy Consumption Prediction Comparison")
plt.xlabel("Samples")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()


# ==============================
# 📌 SAVE MODEL (VERY IMPORTANT)
# ==============================
with open("linear_model.pkl", "wb") as f:
    pickle.dump(lr, f)

print("\n✅ Model saved successfully as linear_model.pkl")

print("\n📁 Saved in:", os.getcwd())