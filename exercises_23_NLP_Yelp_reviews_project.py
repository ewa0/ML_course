import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import string

from pydantic.experimental import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing"
yelp = pd.read_csv(path + "/yelp.csv")
print("Yelp reviews: ", yelp.head())
print("\n", yelp.describe())

yelp["text length"] = yelp["text"].str.len()

# Exploratory Data Analysis
sns.histplot(yelp, x="text length", hue="stars", alpha=0.7, bins=30)
# plt.show()

sns.boxplot(yelp, x="stars", y="text length")
# plt.show()

sns.countplot(yelp, x="stars")
# plt.show()

yelp_gr = yelp.groupby("stars")[["cool", "useful", "funny", "text length"]]
print("\nYelp grouped by stars - average values:\n", yelp_gr.mean())

yelp_corr = yelp_gr.mean().corr()
print("\nYelp correlation matrix:\n", yelp_corr)

sns.heatmap(yelp_corr)
# plt.show()

# ***** NLP Classification ***** #
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)].copy()
# yelp_class["text"] = yelp_class["text"].apply(text_process)
print("\nYelp_class text:\n", yelp_class["text"].head(5))

X = yelp_class['text']
y = yelp_class['stars']

# Count Vectorizer create a matrix of words for the whole set of reviews
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
# print("\nVectorized text data:\n", X)

# ***** Training the dataset ***** #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Naive Bayes Classifier
nb = MultinomialNB()
fit = nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("\nClassification report for Naive Bayesian Classifier:\n", classification_report(y_test, predictions))


# ***** Text Processing ***** #
def text_process(message):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Removing punctuation
    nopunc = [ch for ch in message if ch not in string.punctuation]

    # Joining words into a message
    nopunc = "".join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]


svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', svm_classifier),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

X = yelp_class['text']
y = yelp_class['stars']

# ***** Training the dataset ***** #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# using pipeline to train model
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(
    "\nClassification report for Support Vector Classification (Radial Basis Function (RBF) kernel)  - version 2 with pipeline:\n",
    classification_report(y_test, predictions))
