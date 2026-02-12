import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

sns.set_palette(sns.color_palette("Set2", 8))
husl = sns.husl_palette(8, as_cmap=True)

df = sns.load_dataset('iris')
print("Iris data:\n", df.info(), "\n", df.head(), "\n", df.describe())

# sns.pairplot(data = df, hue="species")
# plt.show()

# Create a kde plot of sepal_length versus sepal width for setosa species of flower.
# setosa = df[df["species"] == "setosa"]
# sns.kdeplot(data=setosa, x="sepal_length", y="sepal_width", fill=True)
# plt.show()

# ***** ***** ***** Support Vector Machine ***** ***** ***** #
# ***** Training the dataset ***** #
y = df["species"]
X = df.drop("species", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

# ***** Evaluating the predictions ***** #
print("\n\nClassification report before optimization:\n", classification_report(y_test, predictions))
print("\n\nConfusion matrix before optimization:\n", confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("support vector machine before optimization")
plt.show()

# ***** Grid search of better parameters ***** #
param_grid = {"C": [0.1, 1.0, 10, 100, 1000], "gamma":[100, 10, 0.1, 0.01]}
grid =  GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
predictions = grid.predict(X_test)
print("\n\nBest parameters from grid search:", grid.best_params_)

# ***** Evaluating the grid predictions ***** #
print("\n\nClassification report after optimization:\n", classification_report(y_test, predictions))
print("\n\nConfusion matrix after optimization:\n", confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("support vector machine after optimization")
plt.show()