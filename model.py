import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

x_train = np.array([43, 21, 25, 42, 57, 59]).reshape(-1,1)
y_train = np.array([99, 65, 79, 75, 87, 81])

# plt.scatter(x_train,y_train)

# plt.show()

model = LinearRegression()
model.fit(x_train,y_train)

r_sq = model.score(x_train,y_train)
print(f"Coefficient of Determination : {r_sq}")

print(f"intercept: {model.intercept_}") #the model predicts 65.14 when x = 0
print(f"slope: {model.coef_}") # predicted response rises by 0.54 when 𝑥 is increased by one.

#input x value to test the model
x_test = np.array([30,35,50]).reshape(-1,1)
y_test = y_test = np.array([80, 72, 90])
y_test_pred = model.predict(x_test)
print(f"predicted response: {y_test_pred}")

#Check the accuracy of the model

r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"The model accuracy using R-squared score is: {r2_test}")
print(f"The model accuracy using mean squared is: {mse_test}")



