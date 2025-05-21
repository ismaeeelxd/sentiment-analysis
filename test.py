import pickle

from helperfun import load_reviews, tokenizer,my_tokenizer


ath = '.\\data\\raw'

text='.\\test\\raw'

review_texts, labels = load_reviews(text)

# with open('shuffled_documents.pkl', 'rb') as f:
#     documents = pickle.load(f)
# X = [d[0] for d in documents]
# Y = [d[1] for d in documents]
# X= review_texts
# Y= labels

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


X_Test_vectorized= vectorizer.transform(review_texts)

filename = './saved_models/model_linear_svm.pkl'
with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)

y_pred = loaded_model.predict(X_Test_vectorized) 
print(y_pred)   