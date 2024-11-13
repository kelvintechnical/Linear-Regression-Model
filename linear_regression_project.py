# Import the numpy library to work with arrays and perform mathematical operations
import numpy as np

# Import matplotlib.pyplot for plotting graphs to visualize data
import matplotlib.pyplot as plt

# Import LinearRegression from sklearn to create our linear regression model
from sklearn.linear_model import LinearRegression



X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # Reshape to make it a 2D array for sklearn

# Create a sample array for y (test scores)
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()

model.fit(X, y)

predicted_score = model.predict([[6]])

# Use the model to predict the score for 6 hours of study
print("Predicted score for 6 hours of study:", predicted_score[0])

plt.scatter(X, y, color='blue', label='Original Data')

plt.plot(X, model.predict(X), color='red', label='Regression Line')

plt.xlabel("Hours Studied")
plt.ylabel("Test Score")


# add a title to the plot 
plt.title("Hours Studied vs. Test Score")


# Show the legend
plt.legend()


#Display the plot
plt.show()