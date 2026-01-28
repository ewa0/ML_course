import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sns.set_palette(sns.color_palette("husl", 8))
husl = sns.husl_palette(8, as_cmap=True)

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors"
df = pd.read_csv(path + "/KNN_Project_Data")
print("KNN_Project_Data:\n", df.info(), df.head(), df.describe())

# sns.pairplot(data=df, hue = "TARGET CLASS")
# plt.show()

scaler =  StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_df = scaler.transform(df.drop('TARGET CLASS',axis=1))
print("transformed data:\n", scaled_df)

# ***** Training the dataset ***** #
# y =  scaled_df["TRAT, GUUB"]
X_train, X_test, y_train, y_test = train_test_split(scaled_df, df["TARGET CLASS"], test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

# ***** Evaluating the predictions ***** #
print("\n\nClassification report:\n", classification_report(y_test, predictions))
print("\n\nConfusion matrix:\n", confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("k = 1")
plt.show()

# ***** Choosing best K value ***** #
error = []
for k in range (1,50):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    error.append(np.mean(predictions != y_test))

plt.scatter(range (1,50), error)
plt.xlabel("k values")
plt.ylabel("error")
plt.xticks((np.arange(1, 50, step=5)), minor = True)
plt.show()

# chosen k = 42
knn = KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
print("\n\nClassification report:\n", classification_report(y_test, predictions))
print("\n\nConfusion matrix:\n", confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions), annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("k = 42")
plt.show()