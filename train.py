import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv('clickbait_dataset.csv', header=None, names=['Headline', 'Clickbait'])

# Convert headlines to lowercase
df['Headline'] = df['Headline'].str.lower()

# Remove stopwords and punctuation
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

df['Headline'] = df['Headline'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words and word not in string.punctuation]))

# Tokenize the text
df['Headline'] = df['Headline'].apply(lambda x: word_tokenize(x))

# Convert tokenized text to string
df['Headline'] = df['Headline'].apply(lambda x: ' '.join(x))

# Extract bag-of-words features
count_vectorizer = CountVectorizer()
bow_features = count_vectorizer.fit_transform(df['Headline'])
# print(bow_features.toarray())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bow_features, df['Clickbait'], test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='1')
recall = recall_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred, pos_label='1')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

joblib.dump(clf, 'clickbait_model.joblib')
joblib.dump(count_vectorizer, 'count_vectorizer.joblib')