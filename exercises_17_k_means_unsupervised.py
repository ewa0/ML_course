import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix

from exercises_13_KNN import predictions

sns.set_palette(sns.color_palette("husl", 8))
husl = sns.husl_palette(8, as_cmap=True)

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/17-K-Means-Clustering"
df = pd.read_csv(path + "/College_Data", index_col=0)
print("\n\nUniversities data:\n", df.info(), df.head(), df.describe())

# ***** ***** ***** Exploratory Data Analysis ***** ***** ***** #
# sns.scatterplot(data = df, x = "Grad.Rate", y = "Room.Board", hue = "Private")
# plt.show()

# sns.scatterplot(data = df, x = "F.Undergrad", y = "Outstate", hue="Private")
# plt.show()

# sns.histplot(data = df, x="Outstate", hue="Private")
# plt.show()

# sns.histplot(data = df, x="Grad.Rate", hue="Private")
# plt.show()

print("\n\nThe name of the school with graduation rate higher than 100%:", df[df["Grad.Rate"] > 100])
df.loc[df["Grad.Rate"] > 100, "Grad.Rate"] = 100

# sns.histplot(data=df, x="Grad.Rate", hue="Private")
# plt.show()

# ***** ***** ***** K Means analysis ***** ***** ***** #
kmeans = KMeans(n_clusters = 2)
X = df.drop("Private", axis = 1)
predictions = kmeans.fit(X)
print("\n\nCluster center vectors:", kmeans.cluster_centers_)

# changing categorical Private to numerical values
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

df["Cluster"] = df["Private"].apply(converter)
print("\n\nConverted df:", df)

# # ***** Evaluating the predictions ***** #
print("\n\nClassification report:\n", classification_report(df["Cluster"], predictions))
print("\n\nConfusion matrix:\n", confusion_matrix(df["Cluster"],predictions))
sns.heatmap(confusion_matrix(df["Cluster"],predictions), annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("K Means")
plt.show()