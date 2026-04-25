from pyexpat.errors import messages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection"
messages = pd.read_csv(path + "/SMSSpamCollection", sep='\t', names=["label", "message"])
print("\n", messages.describe())
print("\n", messages.groupby("label").describe())

messages["length"] = messages["message"].str.len()
print("\n", messages.head())

sns.histplot(data=messages, x="length", hue="label")
# plt.show()

print("\nThe longest text:\n", messages[messages["length"] == 910]["message"].iloc[0])

print("\nExample stop words:\n", stopwords.words('english')[0:10])


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


print("\nTransformed texts, no punctuation and stopwords:\n", messages["message"].head(5).apply(text_process))

# ***** Vectorization ***** #

# Bag of Words transformation
# Count Vectorizer create a matrix of words for the whole set of messages
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print("\nNumber of transformed words from the bag of words:", len(bow_transformer.vocabulary_))

message4 = messages["message"][3]
print("\nMessage number 4:\n", message4)
bow4 = bow_transformer.transform([message4])
print("\nTransformed with a bag of words message number 4:\n", bow4)

print("\nWhich words occur twice:\n", bow_transformer.get_feature_names_out()[4068],
      bow_transformer.get_feature_names_out()[9554])

# Creating Bag of Words for the entire matrix
messages_bow = bow_transformer.transform(messages["message"])
print("\nShape of sparse matrix:\n", messages_bow.shape)
print("\nNumber of non-zero occurences:\n", messages_bow.nnz)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print("\nMatrix sparsity: {}".format(sparsity))

# ***** Term Frequency - Inverse Document Frequency ***** #
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print("\nTF-IDF for message 4:\n", tfidf4)
print("\nTF-IDF for word <u>:", tfidf_transformer.idf_[bow_transformer.vocabulary_["u"]])
print("\nTF-IDF for word <university>:", tfidf_transformer.idf_[bow_transformer.vocabulary_["university"]])

# Applying TF-IDF transformer to the entire Bag of Words matrix
messages_tfidf = tfidf_transformer.transform(messages_bow)
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
print("\npredicted:", spam_detect_model.predict(tfidf4)[0])  # 0 to see the label
# all_pred = spam_detect_model.predict(messages_tfidf)
# print ("\nClassification report for Naive Bayesian Classifier trained on the same data as tested:\n", classification_report(messages['label'], all_pred))

# ***** Training the dataset ***** #
X_train, X_test, y_train, y_test = train_test_split(messages["message"], messages["label"], test_size=0.25)

# Using pipeline to do a few steps with Bag of Words, TF-IDF, and Naive Bayesian Classifier
pipeln = Pipeline([
    ("bow", CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ("tfidf", TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ("classifier", MultinomialNB()),  # train Naive Bayes Classifier, it can be a different classifier
])

# using pipeline to train model
pipeln.fit(X_train, y_train)
predictions = pipeln.predict(X_test)
print("\nClassification report for Naive Bayesian Classifier:\n", classification_report(y_test, pipeln.predict(X_test)))
