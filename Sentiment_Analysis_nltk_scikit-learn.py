
import re
import pickle 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


import nltk

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')



df = pd.read_csv('./movie_data.csv')
df.head(5)

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

# This rounds off the tfidf values to make the array look smoother ar=nd easy to follow. 
np.set_printoptions(precision=2)
# norm=l2 - each output row will have a unit norm. l2 norm is the sum of squares of the vector norms that equals to 1. 
# smoooth_idf = adds 1 to every document seen inorder to avoid diision by 0 to prevent errors. 
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())

# The word 'is' has the highest term frequency. Its tfidf value, instead of being 3, 
# its tfidf value is 0.45. Because it is reallo common across all the documents.
# This explains that it is very unlikely to have any discriminatory information 
# since it is very common across documets and has a very low tfidf value. 

df.loc[0, 'review',][-50:]



def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

preprocessor(df.loc[0, 'review'][-50:])

df['review'] = df['review'].apply(preprocessor)



porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# Un-comment the print statements to reveal the sentiment.
# print("Review: ", X_train[1:2], "\n")
# print("Sentiment: ", y_train[1])

from sklearn.feature_extraction.text import TfidfVectorizer

# lowercase is false since we've aready taken care of it in our preprocessor function
# tokeizer is tokenizer_porter - the one that stemms the words

tfidf = TfidfVectorizer(strip_accents=None, 
                        lowercase=False, 
                        preprocessor=False,
                        tokenizer=tokenizer_porter, 
                        use_idf=True, 
                        norm='l2)',
                        smooth_idf=True)

# y is just the sentiment values
y = df.sentiment.values
# x will be the tfidf representation of our reviews 
X = tfidf.fit_transform(df.review)

# test_size = 0.5 to ensure a 50 50 split.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5,
                                                   shuffle=False)

# cv - cross validations
# max_iter - to have a maximum number of iteratios to ensure that it 
# converges to a particular value, since a logistic function is ever coverging.

# This is training the logistic regression model. 
clf = LogisticRegressionCV(cv=5,
                          scoring='accuracy',
                          random_state=0,
                          n_jobs=-1,
                          verbose=3,
                          max_iter=300)

# saved_model = open('saved_model.sav', 'wb')
# pickle.dump(clf, saved_model)
# saved_model.close()

# filename = 'saved_model.sav'
# saved_clf = pickle.load(open(filename, 'rb'))

# saved_clf.score(X_test, y_test)
