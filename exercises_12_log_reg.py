import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sns.set_palette(sns.color_palette("husl", 8))

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression"
df = pd.read_csv(path + "/advertising.csv")
print("Advertising data:\n", df.info(), df.head(), df.describe())

# sns.histplot(data =df, x =  "Age", bins = 12)
# plt.show()

# sns.jointplot(data = df, x="Age", y="Area Income")
# plt.show()

# sns.jointplot(data = df, x="Age", y="Daily Time Spent on Site", kind="kde",fill=True)
# plt.show()

# sns.jointplot(data=df, x="Daily Internet Usage", y = "Daily Time Spent on Site", kind="scatter")
# plt.show()

# sns.pairplot(data=df, hue="Clicked on Ad")
# plt.show()

# ***** Logistic Regression ***** #
y = df["Clicked on Ad"]
X = df[["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Male"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lg = LogisticRegression(max_iter=100)
lg.fit(X_train, y_train)
# coeff_labeled = pd.DataFrame(lg.coef_, X.columns, columns=["Coefficient"])
# print("\n\nCoefficients:\n", coeff_labeled)

predicted = lg.predict(X_test)
# print("\n\nPredicted length:\n", len(predicted))
# print("\n\ny_test length:\n", len(y_test))
sns.scatterplot(x=y_test, y=predicted)
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.show()

print(classification_report(y_test, predicted))
