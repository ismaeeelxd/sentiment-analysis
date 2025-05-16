import os
import nltk
import string
import pandas as pd
import enum
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
from pathlib import Path
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import  naive_bayes
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.model_selection import train_test_split
# from gensim.models import Word2Vec
import numpy as np
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

tag_map={
    'CC':None,
    'CD':wn.NOUN,
    'DT':wn.NOUN,
    'EX':wn.ADV,
    'FW':None,
    'IN':wn.ADV,
    'JJ':wn.ADJ,
    'JJR':wn.ADJ,
    'JJS':wn.ADJ,
    'LS':None,
    'MD':None,
    'NN':wn.NOUN,
    'NNS':wn.NOUN,
    'NNP':wn.NOUN,
    'NNPS':wn.NOUN,
    'PDT':wn.ADJ,
    'POS':None,
    'PRP':None,
    'PRP$':None,
    'RB':wn.ADV,
    'RBR':wn.ADV,
    'RBS':wn.ADV,
    'RP':wn.ADJ,
    'SYM':None,
    'TO':None,
    'UH':None,
    'VB':wn.VERB,
    'VBD':wn.VERB,
    'VBG':wn.VERB,
    'VBN':wn.VERB,
    'VBP':wn.VERB,
    'VBZ':wn.VERB,
}
stop_words = set(stopwords.words('english'))-{'not'}
stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer() # Create a lemmatizer instance

class Tokenizers(enum.Enum):
    LEMMATIZATION = 1
    STEMMING = 2

def load_reviews(data_path):
    reviews = []
    labels = []
    
    for label_dir, label in [('pos', 1), ('neg', 0)]:
        folder = Path(data_path) / label_dir
        for file_path in folder.glob("*.txt"):
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
                reviews.append(text)
                labels.append(label)

    return reviews, labels

def tokenizer(text, selected_technique):
    text = text.lower()
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)

    processed_tokens = []
    for token, tag in tags:
        if token in stop_words or token in string.punctuation:
            continue

        if selected_technique == Tokenizers.STEMMING:
            processed = stemmer.stem(token)

        else:  # LEMMATIZATION
            wn_tag = tag_map.get(tag)                # could be None
            if wn_tag is not None:
                processed = lemmatizer.lemmatize(token, pos=wn_tag)
            else:
                processed = lemmatizer.lemmatize(token)  # default

        processed_tokens.append(processed)

    return processed_tokens


def vectorize_text(reviews, technique):
    def tokenizer_for_vector(text):
        return tokenizer(text,technique)
    #could use count vectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_for_vector)
    X = vectorizer.fit_transform(reviews)
    return X

data_path = '.\\data\\raw'

# Load and process the data
review_texts, labels = load_reviews(data_path)
print(f"Number of reviews loaded: {len(review_texts)}")
print(f"Number of positive reviews: {sum(labels)}")
print(f"Number of negative reviews: {len(labels) - sum(labels)}")

# Vectorize the reviews
review_texts_vectorized = vectorize_text(review_texts, Tokenizers.LEMMATIZATION)
print(f"Shape of vectorized reviews: {review_texts_vectorized.shape}")


documents = list(zip(review_texts, labels))
random.shuffle(documents)


X = [d[0] for d in documents]
Y = [d[1] for d in documents]

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(tokenizer=lambda text: tokenizer(text, Tokenizers.LEMMATIZATION))
X_Train_vectorized = vectorizer.fit_transform(X_Train)  
X_Test_vectorized = vectorizer.transform(X_Test)        


# LinearSVC
linear_svm = LinearSVC()

linear_svm.fit(X_Train_vectorized, Y_Train)

test_predicted_labels = linear_svm.predict(X_Test_vectorized)

linear_svmAccuracy = accuracy_score(Y_Test, test_predicted_labels)

print(f'Linear SVM Accuracy: {linear_svmAccuracy * 100:.2f}%')

# Logistic Regression
logreg = LogisticRegression()

logreg.fit(X_Train_vectorized, Y_Train)

predicted_labels = logreg.predict(X_Test_vectorized)

LogisticAccuracy = accuracy_score(Y_Test, predicted_labels)

print(f'Logistic Regression accuracy: {LogisticAccuracy * 100:.2f}%')
# Naive Bayes
naive_bayes = naive_bayes.MultinomialNB(alpha=0.6)

naive_bayes.fit(X_Train_vectorized, Y_Train)

test_predicted_labels = naive_bayes.predict(X_Test_vectorized)

naive_bayesAccuracy = accuracy_score(Y_Test, test_predicted_labels)

print(f'Naive Bayes Accuracy: {naive_bayesAccuracy * 100:.2f}%')

svm = SVC(kernel='sigmoid')

svm.fit(X_Train_vectorized, Y_Train)

test_predicted_labels = svm.predict(X_Test_vectorized)

svm_SigACC = accuracy_score(Y_Test, test_predicted_labels)

print(f'SVM Accuracy with sigmoid kernel: {svm_SigACC * 100:.2f}%')

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_Train_vectorized, Y_Train)

rf_predicted_labels = rf.predict(X_Test_vectorized)

rfAccuracy = accuracy_score(Y_Test, rf_predicted_labels)

print(f'Random Forest Accuracy: {rfAccuracy * 100:.2f}%')


labels = ['Linear SVM','SVM (Sigmoid)', 'Logistic Regression',  'Naive Bayes', 'Random Forest']
accuracy = [
    linear_svmAccuracy * 100,
    svm_SigACC * 100,
    LogisticAccuracy * 100,
    naive_bayesAccuracy * 100,
    rfAccuracy * 100
]

plt.figure(figsize=(10, 6))
plt.bar(labels, accuracy, color='skyblue', edgecolor='black')
plt.title('Models Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

