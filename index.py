import pandas as pd
import csv

file = 'dataset_tweet_3.csv'

token_data = open(file)
tokens = csv.reader(token_data, delimiter=';')
tweets = []
label = []
for row in tokens:
    tweets.append(row[0])
    label.append(int(row[1].replace(',','')))

df = pd.DataFrame(columns=['tweets','label'])
df['tweets'] = tweets
df['label'] = label

print (df)


import re,string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

clean_tweets = []
for tweet in tweets:
    def hapus_tanda(tweet): 
        tanda_baca = set(string.punctuation)
        tweet = ''.join(ch for ch in tweet if ch not in tanda_baca)
        return tweet
    
    tweet=tweet.lower()
    tweet = re.sub(r'\\u\w\w\w\w', '', tweet)
    tweet=re.sub(r'http\S+','',tweet)
    #hapus @username
    tweet=re.sub('@[^\s]+','',tweet)
    #hapus #tagger 
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #hapus tanda baca
    tweet=hapus_tanda(tweet)
    #hapus angka dan angka yang berada dalam string 
    tweet=re.sub(r'\w*\d\w*', '',tweet).strip()
    
    #stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tweet = stemmer.stem(tweet)
    clean_tweets.append(tweet)

df['clean'] = clean_tweets
print(df.head())
# print(clean_tweets)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

vectorizer = TfidfVectorizer (max_features=2500)
model_g = GaussianNB()

v_data = vectorizer.fit_transform(df['clean']).toarray()

print (v_data)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(v_data, df['label'], test_size=0.2, random_state=0)
model_g.fit(X_train,y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_preds = model_g.predict(X_test)

print(confusion_matrix(y_test,y_preds))
print(classification_report(y_test,y_preds))
print('nilai akurasinya adalah ',accuracy_score(y_test, y_preds))

tweet = ''
v_data = vectorizer.transform([tweet]).toarray()
y_preds = model_g.predict(v_data)

print(y_preds)
#dengan asumsi bahwa 1 merupakan label positif
if y_preds == 1:
    print('Positif')
else:
    print('Negatif')