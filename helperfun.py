
import string
import enum
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
from pathlib import Path
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
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
stop_words = set(stopwords.words('english'))-{'not', 'no', 'never', 'nor', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'noone', 'nothingness', 'naught', 'naughtiness'}
stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer() # Create a lemmatizer instance

class Tokenizers(enum.Enum):
    LEMMATIZATION = 1
    STEMMING = 2
def my_tokenizer(txt):
    return tokenizer(txt, Tokenizers.LEMMATIZATION)

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
