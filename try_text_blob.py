from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

print(TextBlob("Excellent! Easy to use. Fast Delivery", analyzer=NaiveBayesAnalyzer()).sentiment)
print(TextBlob("Terrible customer service, poor quality, will never buy it again!", analyzer=NaiveBayesAnalyzer()).sentiment)
print(TextBlob("I think the product is okay. It's easy to use. The price is reasonable and you get what you paid for", analyzer=NaiveBayesAnalyzer()).sentiment)



