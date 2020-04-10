import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('imdb_labelled.txt', delimiter = '\t', engine='python', quoting = 3)
df.columns = ["Review", "Result"]

corpus = []
for i in range(0, 999):
  review = re.sub('[^a-zA-Z]', ' ', df["Review"][i])
  review = review.lower()
  review = review.split()
  lemmatizer = WordNetLemmatizer()
  review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values


tf_transformer = TfidfTransformer()
X = tf_transformer.fit_transform(X).toarray()


tfidfVectorizer = TfidfVectorizer(max_features =2000)
X = tfidfVectorizer.fit_transform(corpus).toarray()

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

cm = confusion_matrix(y_test, predictions)
print(cm)

# joblib.dump(tfidfVectorizer, 'tfidfVectorizer.pkl')
# joblib.dump(classifier, 'classifier.pkl')