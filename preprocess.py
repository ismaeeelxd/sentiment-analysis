
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import  naive_bayes
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from helperfun import Tokenizers, load_reviews, my_tokenizer, tokenizer
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

data_path = '.\\data\\raw'

# Load and process the data
review_texts, labels = load_reviews(data_path)
print(f"Number of reviews loaded: {len(review_texts)}")
print(f"Number of positive reviews: {sum(labels)}")
print(f"Number of negative reviews: {len(labels) - sum(labels)}")

# documents = list(zip(review_texts, labels))
# random.shuffle(documents)

# with open('shuffled_documents.pkl', 'wb') as f:
#     pickle.dump(documents, f)

with open('shuffled_documents.pkl', 'rb') as f:
    documents = pickle.load(f)
X = [d[0] for d in documents]
Y = [d[1] for d in documents]
# X= review_texts
# Y= labels
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=labels          # preserves label balance
)



vectorizer = TfidfVectorizer(
    tokenizer=my_tokenizer,
    ngram_range=(1,2),        # unigrams + bigrams
    min_df=3,   
    max_df=0.95,               # ignore terms that appear in less than 3 documents
    analyzer='word',
    sublinear_tf=True,        # use 1 + log(tf)
)
X_Train_vectorized = vectorizer.fit_transform(X_Train)  

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Vectorizer saved successfully.")
X_Test_vectorized = vectorizer.transform(X_Test)        

models = {
    'Linear SVM': LinearSVC(),
    'SVM (Sigmoid)': SVC(kernel='sigmoid'),
    'Logistic Regression': LogisticRegression(verbose=0, max_iter=1000),
    'Naive Bayes': naive_bayes.MultinomialNB(alpha=0.6),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42,bootstrap=False, max_features='sqrt',),
   }

accuracies = {}
train_errors = {}


for name, model in models.items():
    model.fit(X_Train_vectorized, Y_Train)

    test_predictions = model.predict(X_Test_vectorized)
    test_accuracy = accuracy_score(Y_Test, test_predictions) * 100
    accuracies[name] = test_accuracy

    train_predictions = model.predict(X_Train_vectorized)
    train_accuracy = accuracy_score(Y_Train, train_predictions) * 100
    train_error = 100 - train_accuracy
    train_errors[name] = train_error

    print(f'{name} Accuracy on Test Set: {test_accuracy:.2f}%')
    print(f'{name} Accuracy on Train Set: {train_accuracy:.2f}%')
    print(f'{name} Train Error: {train_error:.2f}%\n')
print("////////////////////////////////////////////////////////////\n")

mean_accuracies = {}
from sklearn.model_selection import cross_val_score

for name, model in models.items():
    scores = cross_val_score(model, X_Train_vectorized, Y_Train, cv=10, scoring='accuracy')
    mean_accuracy = np.mean(scores) * 100
    mean_accuracies[name] = mean_accuracy
    print(f'{name} Cross-Validated Accuracy: {mean_accuracy:.2f}%')
    
import seaborn as sns

# Prepare data for plotting
model_names = list(accuracies.keys())
test_acc = [accuracies[name] for name in model_names]
cv_acc = [mean_accuracies[name] for name in model_names]

# Create a DataFrame for visualization
plot_df = pd.DataFrame({
    'Model': model_names * 2,
    'Accuracy': test_acc + cv_acc,
    'Type': ['Test Accuracy'] * len(model_names) + ['Cross-Validation Accuracy'] * len(model_names)
})

# Set up the plot
plt.figure(figsize=(12,6))
sns.barplot(data=plot_df, x='Model', y='Accuracy', hue='Type', palette='Set2')

# Annotate accuracy values on bars
for i, row in plot_df.iterrows():
    plt.text(
        i % len(model_names),
        row['Accuracy'] + 0.3,
        f"{row['Accuracy']:.1f}%",
        ha='center',
        fontsize=9
    )

plt.ylim(70, 100)
plt.title('Model Accuracy: Test Set vs Cross-Validation')
plt.xticks(rotation=45)
plt.ylabel("Accuracy (%)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# with open('tfidf_vectorizer.pkl', 'wb') as f:
#     pickle.dump(vectorizer, f)

# # Save each trained model
# save_dir = './saved_models'
# os.makedirs(save_dir, exist_ok=True)
# for name, model in models.items():
#     filename = os.path.join(save_dir, f'model_{name.replace(" ", "_").lower()}.pkl')
#     with open(filename, 'wb') as f:
#         pickle.dump(model, f)
#     print(f"Saved model to {filename}")

# print("Models, vectorizer, and CV results saved successfully.")