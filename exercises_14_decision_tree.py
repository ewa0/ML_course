import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sns.set_palette(sns.color_palette("husl", 8))
husl = sns.husl_palette(8, as_cmap=True)

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests"
df = pd.read_csv(path + "/loan_data.csv")
print("\n\nLoan data:\n", df.info(), df.head(), df.describe())

# Create a histogram of two FICO distributions
# sns.histplot(data=df,x="fico", hue="credit.policy")
# plt.show()

# sns.histplot(data=df, x="fico", hue="not.fully.paid")
# plt.show()

# sns.countplot(data=df, x="purpose", hue="not.fully.paid")
# plt.show()

# sns.jointplot(data = df, x="fico", y="int.rate", kind="hist")
# plt.show()

# sns.lmplot(data=df, x="fico", y="int.rate", hue="credit.policy", col="not.fully.paid")
# plt.show()

# ***** ***** ***** Decision Tree Model ***** ***** ***** #
# changing categorical purpose to dummies
categorical_features = pd.get_dummies(df["purpose"], columns="purpose", drop_first=True)
print("\nCategorical_features:", categorical_features.head())
final_data = df.join(categorical_features)
final_data.drop("purpose", axis = 1, inplace=True)
print("\n\nFinal data:", final_data.head(), final_data.info())

# ***** Training the dataset ***** #
# X_train, X_test, y_train, y_test = train_test_split(final_data, df["not.fully.paid"], test_size=0.3)
# dtree = DecisionTreeClassifier()
# dtree.fit(X_train,y_train)
# predictions = dtree.predict(X_test)
#
# # ***** Evaluating the predictions ***** #
# print("\n\nClassification report:\n", classification_report(y_test, predictions))
# print("\n\nConfusion matrix:\n", confusion_matrix(y_test,predictions))
# sns.heatmap(confusion_matrix(y_test,predictions), annot=True, fmt="d", cmap="YlGnBu")
# plt.xlabel("predicted")
# plt.ylabel("actual")
# plt.title("decision tree model")
# plt.show()

# ***** ***** ***** Random Forest Classification Model ***** ***** ***** #
# ***** Training the dataset ***** #
X_train, X_test, y_train, y_test = train_test_split(final_data, df["not.fully.paid"], test_size=0.3)
rndforest = RandomForestClassifier()
rndforest.fit(X_train,y_train)
predictions = rndforest.predict(X_test)

# ***** Evaluating the predictions ***** #
print("\n\nClassification report:\n", classification_report(y_test, predictions))
print("\n\nConfusion matrix:\n", confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("random forest")
plt.show()