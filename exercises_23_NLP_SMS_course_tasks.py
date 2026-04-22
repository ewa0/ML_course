import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer

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


print("\nTransformed texts, no puctuation and stopwords:\n", messages["message"].head(5).apply(text_process))

# Bag of Words transformation
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print("\nTransformed words with bag of words:\n", bow_transformer.vocabulary_)


message4 = messages["message"][3]
print("\nMessage number 4:\n", message4)
bow4 = bow_transformer.transform([message4])
print("\nTransformed with a bag of words message number 4:\n", bow4)

print("\nWhich words occur twice:\n", bow_transformer.get_feature_names_out()[4068], bow_transformer.get_feature_names_out()[9554])

# Doing Bag of Words for the entire matrix
messages_bow = bow_transformer.transform(messages["message"])
print("\nShape of sparse matrix:\n", messages_bow.shape)
print("\nNumber of non-zero occurences:\n", messages_bow.nnz)

#TF IDF
