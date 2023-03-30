import nltk

nltk.download("stopwords")
nltk.download("wordnet")


import json
import pandas

import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
print(stopwords.words("english"))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


review_stars = []
review_text = []

with open('yelp_review_part.json', encoding="utf-8") as f:
  for line in f:
    json_line = json.loads(line)
    review_stars.append(json_line["stars"])
    review_text.append(json_line["text"])

dataset = pandas.DataFrame(data={"text": review_text, "stars": review_stars})
print(dataset.shape)


dataset = dataset[0:3000]
dataset = dataset[(dataset['stars']==1)|(dataset['stars']==3)|(dataset['stars']==5)]
print(dataset.shape)


data = dataset['text']
target = dataset['stars']

lemmatizer = WordNetLemmatizer()

def pre_processing(text):
  text_processed = text.translate(str.maketrans('', '', string.punctuation))
  text_processed = text_processed.split()
  result = []
  for word in text_processed:
    word_processed = word.lower()
    if word_processed not in stopwords.words("english"):
      word_processed = lemmatizer.lemmatize(word_processed)
      result.append(word_processed)
  return result
  
count_vectorize_tranformer = CountVectorizer(analyzer=pre_processing).fit(data)

data = count_vectorize_tranformer.transform(data)

print(data)


machine = MultinomialNB()
machine.fit(data,target)


new_reviews = pandas.read_csv("new_reviews.csv", header=None)
new_reviews_transformed = count_vectorize_tranformer.transform(new_reviews.iloc[:,0])

prediction = machine.predict(new_reviews_transformed)
prediction_prob = machine.predict_proba(new_reviews_transformed)

print(prediction)
print(prediction_prob)


new_reviews['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)

new_reviews = pandas.concat([new_reviews,prediction_prob_dataframe], axis=1)

new_reviews.to_csv("new_reviews_with_prediction.csv")










