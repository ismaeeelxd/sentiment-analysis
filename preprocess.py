import nltk
import string
import enum
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
from pathlib import Path

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


stop_words = stopwords.words('english')
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
    #should add stemming or lemmatization later
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and token not in string.punctuation:
            word = token

            if selected_technique == Tokenizers.STEMMING:
                word = stemmer.stem(word)
            elif selected_technique == Tokenizers.LEMMATIZATION:
                word = lemmatizer.lemmatize(word)

            processed_tokens.append(word)
    return processed_tokens


    

def vectorize_text(reviews, technique):
    def tokenizer_for_vector(text):
        return tokenizer(text,technique)
    #could use count vectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_for_vector)
    X = vectorizer.fit_transform(reviews)
    return X

data_path = '.\\data\\raw'

# X is a huge array containing the reviews and Y is a huge array containing the corresponding label
review_texts,labels = load_reviews(data_path)
review_texts_vectorized = vectorize_text(review_texts,Tokenizers.STEMMING)
