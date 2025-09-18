import nltk
import random
from nltk.corpus import movie_reviews

try:
    nltk.data.find('corpora/movie_reviews')
except LookupError:
    print("[INFO] movie_reviews not found. Downloading...")
    nltk.download('movie_reviews')

# A total of 1,583,820 words
# Two categories pos and neg
# Field ids are 2000
# 39768 the total number of distinct words in ‘movie_reviews’


documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

texts = []
labels = []

for words, label in documents:
    texts.append(" ".join(words))

for words, label in documents:
    labels.append(label)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train_counts, y_train)

# from sklearn.metrics import accuracy_score, classification_report

y_pred = clf.predict(X_test_counts)
# print("Accurancy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

review_text = input("Enter a movie review: ")

review_list = [review_text]

review_counts = vectorizer.transform(review_list)
prediction = clf.predict(review_counts)

print(f"\nYour review sentiment: {prediction[0]}")