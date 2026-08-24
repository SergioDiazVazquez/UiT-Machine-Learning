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

# 5. Plot the data

# 1. Plot the actual data points
plt.scatter(X, y, color='blue', label='Actual Temperatures')

# 2. Plot the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

# 3. Add labels and a legend
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Temperature vs. Years')
plt.legend()

# 4. Show the graph
plt.show()

# Problem 2b
# R2 tell you how well your line fits the data
print("The score of our model is:", model.score(X, y))