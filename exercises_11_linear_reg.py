import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

sns.set_palette(sns.color_palette("muted"))

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression"
df = pd.read_csv(path + "/Ecommerce Customers", sep=",")
print("Ecommerce Customers data:\n", df.info(), df.head(), df.describe())

# sns.jointplot(data = df, x = "Time on Website", y = "Yearly Amount Spent")
# plt.show()
# sns.heatmap(data=df[["Time on Website","Yearly Amount Spent"]].corr(), annot=True)
# plt.show()

# sns.jointplot(data = df, x = "Time on App", y = "Yearly Amount Spent")
# plt.show()
# sns.heatmap(data=df[["Time on App","Yearly Amount Spent"]].corr(), annot=True)
# plt.show()

# sns.jointplot(data = df, x = "Time on App", y = "Length of Membership", kind="hex")
# plt.show()

# sns.pairplot(data=df)
# plt.tight_layout()
# plt.show()

# sns.lmplot(data = df, x="Yearly Amount Spent", y="Length of Membership")
# plt.show()

# ***** training the model ***** #
X = df[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = df["Yearly Amount Spent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)
coeff_labeled = pd.DataFrame(lm.coef_, X.columns, columns=["Coefficient"])
print("\n\nCoefficients:\n",coeff_labeled)

predicted = lm.predict(X_test)
print("\n\nPredicted length:\n", len(predicted))
print("\n\ny_test length:\n", len(y_test))
sns.scatterplot(x = y_test, y = predicted)
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.show()

# ***** evaluating the model ***** #
MAE = metrics.mean_absolute_error(y_test, predicted)
MSE = metrics.mean_squared_error(y_test, predicted)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, predicted))
print("\n\nMean Average Error:", MAE)
print("Mean Square Error:", MSE)
print("Root Mean Square Error:", RMSE)

plt.hist(y_test - predicted, bins=30)
plt.xlabel("Residuals")
plt.ylabel("Count")
plt.show()