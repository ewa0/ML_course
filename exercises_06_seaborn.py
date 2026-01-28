import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
# colors = ["#FFBC42", "#FFBC42", "#FFBC42", "#8F2D56", "#8F2D56"]
sns.set_palette(sns.color_palette("pastel"))
titanic = sns.load_dataset('titanic')
print(titanic.head())
sns.jointplot(data=titanic, x="fare", y="age")
plt.show()

sns.histplot(data=titanic, x="fare", color="red", bins=30)
plt.show()

sns.boxplot(data=titanic, x="class", y="age", hue="pclass")
plt.show()

sns.swarmplot(data=titanic, x="class", y="age", hue="pclass")
plt.show()

sns.histplot(data=titanic, x="sex", hue="sex")
plt.show()

correlation=titanic[["survived", "pclass", "age", "sibsp", "parch", "fare", "adult_male", "alone"]].corr()
sns.heatmap(data=correlation)
plt.show()

fg= sns.FacetGrid(data=titanic, col="sex")
fg.map(sns.histplot, "age")
plt.show()