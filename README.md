<!DOCTYPE html>
<html>
<head>
  <title>Linear Regression Model: Predicting Test Scores Based on Study Hours</title>
</head>
<body>

<h1>Linear Regression Model: Predicting Test Scores Based on Study Hours</h1>

<h2>Project Overview</h2>
<p>This project demonstrates the application of a <strong>Linear Regression Model</strong> to predict test scores based on the hours studied. Linear regression is one of the most fundamental machine learning algorithms, and building this project offers a solid introduction to machine learning. In this project, we use Python libraries like NumPy, Matplotlib, and scikit-learn to build, train, and visualize our model.</p>

<h2>What I Learned</h2>
<p>In building this project, I learned the following:</p>
<ul>
  <li><strong>Data Manipulation</strong>: How to use NumPy arrays to structure data for machine learning models.</li>
  <li><strong>Model Training</strong>: How to apply scikit-learn’s <code>LinearRegression</code> to train a model with simple, small datasets.</li>
  <li><strong>Data Visualization</strong>: How to use Matplotlib to plot both the original data and the regression line, helping to visualize the relationship between study hours and test scores.</li>
  <li><strong>Model Prediction</strong>: How to use a trained model to make predictions, such as predicting a test score based on new input hours.</li>
</ul>

<h2>Why This Project is Important</h2>
<p>Understanding and implementing linear regression is essential for machine learning and data science beginners. It serves as a stepping stone to more complex machine learning algorithms and concepts. Linear regression also provides insight into how data points relate to each other, allowing us to make informed predictions.</p>

<h2>Why This Project is Great for a Machine Learning Portfolio</h2>
<p>Linear regression is widely used in various industries for predictive analysis. By showcasing this project, I can demonstrate my knowledge of machine learning basics and my ability to apply that knowledge to real-world data. This project is a fantastic foundation for more advanced machine learning projects and techniques.</p>

<h2>Code Walkthrough</h2>

<pre><code># Importing necessary libraries
import numpy as np                     # For handling arrays and performing math operations
import matplotlib.pyplot as plt         # For plotting graphs
from sklearn.linear_model import LinearRegression  # For creating the linear regression model

# Data Preparation
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable: hours studied
y = np.array([2, 4, 5, 4, 5])                 # Dependent variable: test scores

# Model Initialization and Training
model = LinearRegression()              # Creating the model
model.fit(X, y)                         # Training the model with the data

# Making Predictions
predicted_score = model.predict([[6]])  # Predicting the score for 6 hours of study
print("Predicted score for 6 hours of study:", predicted_score[0])

# Data Visualization
plt.scatter(X, y, color='blue', label='Original Data')    # Plotting original data points
plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Plotting the regression line
plt.xlabel("Hours Studied")                               # X-axis label
plt.ylabel("Test Score")                                  # Y-axis label
plt.title("Hours Studied vs. Test Score")                 # Plot title
plt.legend()                                              # Showing the legend
plt.show()                                                # Display the plot
</code></pre>

<h2>Follow Me</h2>
<p>Stay connected with my latest projects and insights:</p>
<ul>
  <li><strong>Bluesky</strong>: <a href="https://bsky.app/profile/kelvintechnical.bsky.social">kelvintechnical.bsky.social</a></li>
  <li><strong>X (formerly Twitter)</strong>: <a href="https://x.com/kelvintechnical">kelvintechnical</a></li>
  <li><strong>LinkedIn</strong>: <a href="https://www.linkedin.com/in/kelvin-r-tobias-211949219/">Kelvin R. Tobias</a></li>
</ul>

</body>
</html>
