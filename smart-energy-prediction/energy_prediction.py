import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ===============================
# LOAD DATA (AUTO FIX)
# ===============================

df = pd.read_csv("energy_data.csv", header=None)

# split single column into multiple columns
df = df[0].str.split(",", expand=True)

# set column names
df.columns = ["date","time","temperature","humidity","energy"]

# remove header row
df = df.iloc[1:]

# convert numeric columns
df["temperature"] = df["temperature"].astype(float)
df["humidity"] = df["humidity"].astype(float)
df["energy"] = df["energy"].astype(float)

print(df.head())


# ===============================
# FEATURE ENGINEERING
# ===============================

df["hour"] = pd.to_datetime(df["time"]).dt.hour
df["day"] = pd.to_datetime(df["date"]).dt.day
df["month"] = pd.to_datetime(df["date"]).dt.month


# ===============================
# FEATURES
# ===============================

X = df[["hour","day","month","temperature","humidity"]]
y = df["energy"]


# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# SCALING
# ===============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# LINEAR REGRESSION
# ===============================

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)


# ===============================
# RANDOM FOREST
# ===============================

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# ===============================
# RESULTS
# ===============================

print("\nLinear Regression")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("R2:", r2_score(y_test, lr_pred))


print("\nRandom Forest")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("R2:", r2_score(y_test, rf_pred))


# ===============================
# GRAPH
# ===============================

plt.plot(y_test.values, label="Actual")
plt.plot(lr_pred, label="Linear")
plt.plot(rf_pred, label="RandomForest")

plt.legend()
plt.show()


# ===============================
# NEW PREDICTION
# ===============================

new_data = [[17,1,1,29,27]]

new_data = scaler.transform(new_data)

print("\nPrediction")
print("Linear:", lr.predict(new_data)[0])
print("RandomForest:", rf.predict(new_data)[0])