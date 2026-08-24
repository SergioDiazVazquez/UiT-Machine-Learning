import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load the data
df = pd.read_csv("exercise-2/global-temperatures.csv", sep=r"\s+")

X = df[["Year"]]
y = df["Temperature"]

# 2. Model
model = LinearRegression()

# 3. Train it on the dataset
model.fit(X,y)

# 4. Generate the predicted temperatures
y_pred = model.predict(X)

# 5. Plot the residuals

# 1. Calculate the residuals
residuals = y - y_pred

# 2. Create the residual plot
plt.figure(figsize=(8, 5))
plt.scatter(X, residuals, color='purple', label='Residuals')

# 3. Add a horizontal line at 0 for reference
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)

# 4. Add labels and title
plt.xlabel('Year')
plt.ylabel('Residuals (Observed - Predicted)')
plt.title('Residual Plot for Temperature Model')
plt.legend()

# 5. Show the plot
plt.show()