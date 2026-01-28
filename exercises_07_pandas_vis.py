import matplotlib.pyplot as plt

plt.style.use('ggplot')
import pandas as pd

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz"
df3 = pd.read_csv(path + "/df3")
print(df3.head())
df3.plot.scatter(x="a", y="b", c="red", s = 16, figsize = (12,3))
plt.show()

df3["a"].plot.hist(bins=30)
plt.show()

df3[["a","b"]].plot.box()
plt.show()

df3["d"].plot.kde(lw=3, ls='--')
plt.show()

ax = df3[:30].plot.area(alpha=0.4, legend=False)
fig = ax.figure
fig.legend(bbox_to_anchor=(1.01, 0.65))
plt.show()