import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score

from imblearn.over_sampling import RandomOverSampler

# Load data
data = pd.read_csv("C:\\Users\\SURYA VARMA\\OneDrive\\Desktop\\SMS SPAN DETECTION\\SMS SPAM\\spam.csv", encoding="latin-1")

# Data preprocessing
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.rename(columns={'v1': 'Target', 'v2': 'Text'}, inplace=True)

# Visualizations
cols = ["#05A8AA", "#EF233C"]
plt.figure(figsize=(12, 8))
fg = sns.countplot(x=data["Target"], palette=cols)
fg.set_title("Count of Spam and Ham")
fg.set_xlabel("Classes")
fg.set_ylabel("Count")
plt.show()

data["Text_Length"] = data["Text"].apply(len)

# Clean text
def clean(text):
    sms = re.sub('[^a-zA-Z]', " ", text)
    sms = sms.lower()
    sms = sms.split()
    sms = ' '.join(sms)
    return sms

data["Cleaned_text"] = data["Text"].apply(clean)

# Tokenize text
data["Tokenized_text"] = data.apply(lambda row: nltk.word_tokenize(row["Cleaned_text"]), axis=1)

# Remove stopwords
nltk.download('stopwords')
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

data["Nostopword_text"] = data["Tokenized_text"].apply(remove_stopwords)

# Stemming
nltk.download('wordnet')
stemmer = PorterStemmer()
def stem_word(text):
    stems = [stemmer.stem(word) for word in text]
    return stems

data["Stemmed_text"] = data["Nostopword_text"].apply(stem_word)

# Create corpus
corpus = []
for i in data["Stemmed_text"]:
    msg = ' '.join([row for row in i])
    corpus.append(msg)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()

# Encode target variable
label_encoder = LabelEncoder()
data["Target"] = label_encoder.fit_transform(data["Target"])
y = data["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Oversampling
oversampler = RandomOverSampler(random_state=11)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# Visualize class distribution after oversampling
resampled_df = pd.DataFrame({'Target': y_train})
plt.figure(figsize=(8, 6))
sns.countplot(x='Target', data=resampled_df, palette=cols)
plt.title('Class Distribution After Oversampling')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Classifier
classifier = MultinomialNB()

# Model training and cross-validation
cv_score = cross_val_score(classifier, X_train, y_train, scoring="accuracy", cv=10)
print("NaiveBayes: %f " % (cv_score.mean()))

# Evaluation metrics
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix of NaiveBayes')
plt.show()

print(classification_report(y_test, y_pred))

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (fn + tp)
f1 = 2 * (precision * recall) / (precision + recall)
sensitivity = tp / (fn + tp)
specificity = tn / (tn + fp)
fpr = fp / (tn + fp)
fnr = fn / (fn + tp)
npv = tn / (tn + fn)
fdr = fp / (fp + tp)
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

metrics_df = pd.DataFrame({
    'Model': ["NaiveBayes"],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Measure': [f1],
    'Sensitivity': [sensitivity],
    'Specificity': [specificity],
    'FPR': [fpr],
    'FNR': [fnr],
    'NPV': [npv],
    'FDR': [fdr],
    'MCC': [mcc]
})

print(metrics_df.transpose())

