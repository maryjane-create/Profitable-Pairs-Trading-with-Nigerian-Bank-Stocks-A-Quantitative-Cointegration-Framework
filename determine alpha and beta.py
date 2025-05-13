import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt



# Load CSVs (ZENITH and UBA should be aligned by date)
zenith = pd.read_csv("/content/NSENG_ZENITHBANK, 1D_6d9c1.csv", parse_dates=["time"], index_col="time")
uba = pd.read_csv("/content/NSENG_UBA, 1D_d298c.csv", parse_dates=["time"], index_col="time")

# Extract log prices
zenith["log_price"] = np.log(zenith["close"])
uba["log_price"] = np.log(uba["close"])

# Merge the log prices and ensure aligned indices
data = pd.DataFrame({
    "log_ZENITH": zenith["log_price"],
    "log_UBA": uba["log_price"]
}).dropna()  # Drop rows with missing values to align indices

X = sm.add_constant(data["log_UBA"])
model = sm.OLS(data["log_ZENITH"], X).fit()
beta = model.params["log_UBA"]
alpha = model.params["const"]
print(f"β = {beta:.4f}, α = {alpha:.4f}")
data["spread"] = data["log_ZENITH"] - beta * data["log_UBA"]

# Static Z-score (based on entire history)
mean_spread = data["spread"].mean()
std_spread = data["spread"].std()
data["zscore"] = (data["spread"] - mean_spread) / std_spread

# Optional: Rolling Z-score (e.g., 60-day window)
# data["zscore_rolling"] = (data["spread"] - data["spread"].rolling(60).mean()) / data["spread"].rolling(60).std()

# Example: GTCO and UBA log prices
X = zenith["log_price"]
Y = uba["log_price"]

# Ensure X and Y have the same index before regression
# Reindex X and Y to the intersection of their indices
common_index = X.index.intersection(Y.index)
X = X.reindex(common_index)
Y = Y.reindex(common_index)


# Regression 1: GTCO ~ UBA
X1 = sm.add_constant(X)
model1 = sm.OLS(Y, X1).fit()
resid1 = model1.resid
std1 = np.std(resid1)

# Regression 2: UBA ~ GTCO
X2 = sm.add_constant(Y)
model2 = sm.OLS(X, X2).fit()
resid2 = model2.resid
std2 = np.std(resid2)

if std1 < std2:
    print("zenith is the pacesetter (independent variable)")
else:
    print("uba is the pacesetter (independent variable)")



plt.figure(figsize=(14, 6))
plt.plot(data.index, data["zscore"], label="Z-score")
plt.axhline(0, color="black")
plt.axhline(1.0, color="red", linestyle="--", label="Sell Signal")
plt.axhline(-1.0, color="green", linestyle="--", label="Buy Signal")
plt.title("Z-Score of Spread between ZENITHBANK and UBA")
plt.legend()
plt.grid(True)
plt.show()
