from itertools import groupby

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("QtAgg")
import seaborn as sns

sns.set_palette(sns.color_palette("muted"))

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects"
df = pd.read_csv(path + "/911.csv")
print("911 data:\n", df.info(), df.head())


print("\nTop 5 zipcodes for 911 calls:\n", df['zip'].value_counts().head(5))
print("\nTop 5 townships for 911 calls:\n", df['twp'].value_counts().head(5))
print("\nUnique title codes for 911 calls:\n", df['title'].nunique())

# Create 'Reason' column
df["Reason"] = df["title"].apply(lambda reason: reason.split(":")[0])
print(df.head())
print("\nThe most common reason for calls:\n", df["Reason"].value_counts().head())
sns.countplot(df, x = "Reason")
plt.show()

print("\nData type of 'timeStamp' column:\n", df['timeStamp'].dtypes)
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
print("\nNew data type of 'timeStamp' column:\n", df['timeStamp'].dtypes)
time = df['timeStamp'].dt
df["hour"] = time.hour
df["month"] = time.month
df["day of week"] = time.dayofweek
df["date"] = time.date
df["day of week"] = df["day of week"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
print("\nData with new columns:\n", df["day of week"].head())

sns.countplot(data=df, x="day of week", hue="Reason")
plt.show()

sns.countplot(data=df, x="month", hue="Reason")
plt.show()

byMonth = df.groupby("month").count()
print("\n911 calls by month:\n", byMonth.head())
sns.lineplot(data=byMonth, x="month", y="e")
sns.lmplot(data=byMonth.reset_index(), x="month", y="e")
plt.show()

byDate = df.groupby("date").count()
print("byDate:\n", byDate)
df["reason_date_count"] = df.groupby(["Reason", "date"])["e"].transform("count")
sns.FacetGrid(df, row="Reason", height=3, aspect=2.5).map(sns.lineplot, "date", "reason_date_count").add_legend()
plt.show()

# ***** day of week vs. hour heatmap *****
indexed_df = df.set_index(["day of week", "hour"])
unstacked_df = indexed_df.groupby(by=["day of week","hour"]).count()["e"].unstack()
print("\n\n\nData ready for a heatmap with day of week as a column:\n", unstacked_df)

sns.heatmap(unstacked_df, cmap="PiYG")
plt.show()

sns.clustermap(unstacked_df, cmap="PiYG")
plt.show()

# ***** day of week vs. hour heatmap *****
indexed_df = df.set_index(["day of week", "month"])
unstacked_df = indexed_df.groupby(by=["day of week","month"]).count()["e"].unstack()
print("\n\n\nData ready for a heatmap with month as a column name:\n", unstacked_df)

sns.heatmap(unstacked_df, cmap="twilight_shifted")
plt.show()

sns.clustermap(unstacked_df, cmap="twilight_shifted")
plt.show()