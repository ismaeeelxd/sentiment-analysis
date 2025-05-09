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
# from gensim.models import Word2Vec
import numpy as np
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

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
