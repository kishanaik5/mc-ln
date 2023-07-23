#9
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

text_data = [
    "I love this sandwich, pos",
    "This is an amazing place, pos",
    "I feel very good about these cheese, pos",
    "This is my best work, pos",
    "What an awesome view, pos",
    "I do not like this restaurant, neg",
    "I am tired of this stuff, neg",
    "I can't deal with this, neg",
    "He is my sworn enemy, neg",
    "My boss is horrible, neg",
    "This is an awesome place, pos",
    "I do not like the taste of this juice, neg",
    "I love to dance, pos",
    "I am sick and tired of this place, neg",
    "What a great holiday, pos",
    "That is a bad locality to stay, neg",
    "We will have good fun tomorrow, pos",
    "I went to my enemy's house today, neg"
]

labels = ['pos', 'pos', 'pos', 'pos', 'pos', 'neg', 'neg', 'neg', 'neg', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg']

df = pd.DataFrame({'text': text_data, 'label': labels})


X = df['text']
y = df['label']


vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)


classifier = MultinomialNB()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label='pos')
precision = precision_score(y_test, y_pred, pos_label='pos')
confusion_mat = confusion_matrix(y_test, y_pred)

print("Total Instances of Dataset:", len(df))
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("Confusion Matrix:")
print(confusion_mat)
