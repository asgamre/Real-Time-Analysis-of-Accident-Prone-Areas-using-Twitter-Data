from tweepy import Stream
from tweepy.streaming import StreamListener
import pandas as pd
import Credentials
from Preprocess import preprocess
from Classifier import train,predictInterest,testClassifier,printMetrics
targetWords = ['crash', 'accident', 'wreck','mishap']
label = {'S':'Situational','N':'Non-Situational'}

class listener(StreamListener):
    def __init__(self):
        global SVMClassifier, NBClassifier, bestCount_Vectorizer
        super().__init__()
        print("Initializing")
        trainingSet = pd.read_csv("dataset/train.csv", usecols=['Tweet', 'Label'])
        testSet = pd.read_csv("dataset/test.csv", usecols=['Tweet', 'Label'])
        print("Initiating Training...")
        print()
        NBClassifier, bestCount_Vectorizer, best_tfidf, interestLabels = train(trainingSet)
        print()
        print("Initiating Testing...")
        print()
        testClassifier(NBClassifier, testSet, bestCount_Vectorizer)
        printMetrics(NBClassifier, best_tfidf, interestLabels, "NBClassifier")
        print()
        print("Starting Live tweets")

    def on_status(self, status):
        tweet = str(status.text).lower()
        if any(i in tweet for i in targetWords):
            preprocessedtweet = preprocess(tweet)
            predictedlabel = predictInterest([preprocessedtweet], NBClassifier, bestCount_Vectorizer)
            print(tweet)
            print(label[predictedlabel[0]])

    def on_error(self, status_code):
        if status_code == 420:
            return False


auth = Credentials.authenticate()
twitterStream = Stream(auth, listener())
twitterStream.filter(locations=[-97.318268,32.760717,-96.600723,33.207095],languages = ["en"], stall_warnings = True)
