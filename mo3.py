
# Project M03: Car acceleration time from horsepower

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# STEP 1: Load the data file (it must be in the same folder)
df = pd.read_csv("auto-mpg.csv")
print("Rows and columns:", df.shape)

# STEP 2: Clean the data
# horsepower has "?" for 6 cars -> turn it into a number (bad values become empty)
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower", "acceleration"])

# STEP 3: Choose input X and output y
X = df[["horsepower"]]       # double brackets: X must be a table
y = df["acceleration"]

# STEP 4: Split: 80% for train, 20% for test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 5: Train the model (find the best straight line)
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 6: Test on the hidden 20%
y_pred = model.predict(X_test)
print("R2 score       :", round(r2_score(y_test, y_pred), 3))
print("Average error  :", round(mean_absolute_error(y_test, y_pred), 2), "s")
print("Slope m        :", round(model.coef_[0], 4))
print("Intercept c    :", round(model.intercept_, 2))

# STEP 7: Predict a new value and draw the graph
new = pd.DataFrame({"horsepower": [150]})
print("Predicted 0-60 mph time for 150 hp:", round(model.predict(new)[0], 2), "s")

line_x = pd.DataFrame({"horsepower": sorted(X["horsepower"])})
plt.scatter(X["horsepower"], y, s=12, alpha=0.6, label="Actual (test data)")
plt.plot(line_x["horsepower"], model.predict(line_x), color="red", label="Regression line")
plt.xlabel("Engine power (horsepower)")
plt.ylabel("Time for 0-60 mph (seconds)")
plt.title("Car acceleration time from horsepower")
plt.legend()
plt.show()
