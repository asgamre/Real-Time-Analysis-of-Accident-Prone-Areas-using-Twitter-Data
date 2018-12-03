import pandas as pd
from Preprocess import preprocess
from Connector import targetWords

dataset=open("dataset/tweetset.csv","w+")

data = pd.read_csv("corpus/corpusDallas.csv", sep=';', usecols=['text'])
for tweet in data.values:
    tweet = str(tweet[0]).lower()
    if any(i in tweet for i in targetWords):
        dataset.write('{}\n'.format(preprocess(tweet)))


