import datetime
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

Count_Vectorizer, porter= None, None

# defines the required global variables
def initialize(i,j):
    global Count_Vectorizer, porter
    print("Ngram range {},{}".format(i,j))
    Count_Vectorizer = CountVectorizer(ngram_range=(i, j))  # change settings for unigram, bigram, trigram
    porter = PorterStemmer()

# takes in a pandas dataframe and returns a feature vector and list of labels
def getFeatureVectorAndLabels(dataframe):
    print()
    print("Converting dataframe to list of tweets and list of interests")
    listOfTweets, labels = dfToTweetsAndLabels(dataframe)
    print("Stemming list of tweets")
    listOfTweets = stemList(porter, listOfTweets)
    data_counts = Count_Vectorizer.fit_transform(listOfTweets)
    print("There are %s features" % data_counts.shape[1])
    print()
    tfidf_doc = TfidfTransformer(use_idf=True).fit_transform(data_counts)
    return tfidf_doc, labels

# takes in a stemmer object defined in initalize() and a list of strings to be stemmed
def stemList(stemmer, listOfTweets):
    if stemmer != None:
        stemmedTokens = []
        for sentence in listOfTweets:
            tokens = sentence.split(' ')
            tokens = [stemmer.stem(token) for token in tokens if not token.isdigit()]
            stemmedTokens.append(tokens)
        listOfTweets = []
        for token in stemmedTokens:
            listOfTweets.append(" ".join(str(i) for i in token))
    return listOfTweets

# Returns a list of tweets and labels
def dfToTweetsAndLabels(dataframe):
    listOfTweets = []
    interestLabels = []
    for i in dataframe.index:
        tweet = dataframe["Tweet"][i]
        interest = dataframe["Label"][i]
        listOfTweets.append(tweet)
        interestLabels.append(interest)
    return listOfTweets, interestLabels

# returns an interest prediction
def predictInterest(tweet,classifier,bestCount_Vectorizer):
    data_counts = bestCount_Vectorizer.transform(tweet)
    tfidf_doc = TfidfTransformer(use_idf=True).fit(data_counts).transform(data_counts)
    prediction = classifier.predict(tfidf_doc)
    return prediction

# uses the cross_val_score method to calculate the accuracy of a model using kfold cross validation, with cv being the number of folds
def printKFoldScore(classifier, features, labels, name):
    kfold_score = cross_val_score(classifier, features, labels, cv=10)
    return kfold_score.mean()

# takes a prediction using the desired classifier and prints the classification report and confusion matrix
def printMetrics(classifier, features, labels, name):
    predictedList = classifier.predict(features)
    print()
    print("Classification report for " + name)
    print(classification_report(labels, predictedList))
    cm = confusion_matrix(labels, predictedList)
    print("Confusion Matrix")
    print(cm)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # np.set_printoptions(precision=2)
    # plt.figure()
    # plotMatrix(cm)
    # plt.show()


def plotMatrix(cm, title='Confusion matrix', cmap=plt.cm.YlOrRd):
    target_names = ['non-situational','situational']  # alphabetical order
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# runs a classifier against a manually curated dataset and calculates the accuracy
def testClassifier(classifier, dataframe,bestCount_Vectorizer):
    print("N-gram Range:".format(bestCount_Vectorizer.ngram_range))
    interestLabels, predictedInterests, listOfTweets = [], [], []
    print("Running on test set")
    for i in dataframe.index:
        tweet = dataframe["Tweet"][i]
        interest = dataframe["Label"][i]
        interestLabels.append(interest)
        predictedInterests.append(predictInterest([tweet], classifier,bestCount_Vectorizer))
        listOfTweets.append(tweet)
    if len(interestLabels) == len(predictedInterests):
        correct = 0
        print("False predictions:")
        for j in range(len(interestLabels)):
            if interestLabels[j] == predictedInterests[j]:
                correct += 1
        print("{}% accuracy".format(correct/len(interestLabels)))


# train multinomial naive bayes classifier with given features and labels
def trainNB(features,labels):
	starttime = datetime.datetime.now()
	clf = MultinomialNB().fit(features, labels)
	print("Time taken to train NBClassifier: " + str(datetime.datetime.now() - starttime))
	return clf


# takes in the training set and returns a trained object and the best ngram feature object
def train(trainingSet):
    print("Initializing variables and environment")
    ngrams = [(1,1),(2,2),(3,3),(1,2),(1,3)]
    best_accuracyNB = None
    best_NBClassifier = None
    best_tfidf = None
    bestCount_Vectorizer = None
    for (i,j) in ngrams:
        initialize(i,j)
        tfidf_doc, interestLabels = getFeatureVectorAndLabels(trainingSet)
        loadFromSave = False
        if loadFromSave:
            # joblib is an sklearn library that allows us to save / load the trained classifiers
            NBClassifier = joblib.load('classifiers/naivebayes.pkl')
            print("Classifiers loaded from file")
        else:
            NBClassifier = trainNB(tfidf_doc, interestLabels)
            joblib.dump(NBClassifier, 'classifiers/naivebayes.pkl')

        accuracyNB = printKFoldScore(NBClassifier, tfidf_doc, interestLabels, "NBClassifier")
        print("Accuracy for NBClassifier: " + str(accuracyNB))
        printMetrics(NBClassifier, tfidf_doc, interestLabels, "NBClassifier")

        if not best_accuracyNB:
            best_accuracyNB = accuracyNB
            best_NBClassifier = NBClassifier
            bestCount_Vectorizer = Count_Vectorizer
            best_tfidf = tfidf_doc
        if accuracyNB > best_accuracyNB:
            best_accuracyNB = accuracyNB
            best_NBClassifier = NBClassifier
            bestCount_Vectorizer = Count_Vectorizer
            best_tfidf = tfidf_doc

    return best_NBClassifier,bestCount_Vectorizer,best_tfidf,interestLabels
    # return NBClassifier,Count_Vectorizer
