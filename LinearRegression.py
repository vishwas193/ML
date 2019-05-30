import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,r2_score

diabetes=datasets.load_diabetes()

#using only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
print(len(diabetes_X))

diabetes_X_train=diabetes_X[:-20]
diabetes_X_test=diabetes_X[-20:]

diabetes_y_train=diabetes.target[:-20]
diabetes_y_test=diabetes.target[-20:]

#create Linear Regresion Object
regr=linear_model.LinearRegression()

#training the model
regr.fit(diabetes_X_train,diabetes_y_train)

diabetes_y_pred=regr.predict(diabetes_X_test)

#Coefficients
print("Regression Coefficients",regr.coef_)
print("Intercept", regr.intercept_)

#Mean squared Error
print("MSE", mean_squared_error(diabetes_y_test,diabetes_y_pred))

#Variance Score: 1 is the perfect prediction
print('Variance Score (1 is perfect perfect prediction):',r2_score(diabetes_y_test,diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
