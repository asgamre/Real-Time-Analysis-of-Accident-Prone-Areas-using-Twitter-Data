import re, datetime
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

Count_Vectorizer, Tfidf_Transformer, cachedStopwords, porter= None, None, None, None


# defines the required global variables, as well as stopwords or stemmers required
def initializeCV(i,j):
    global Count_Vectorizer, cachedStopwords, porter
    print("Ngram range {},{}".format(i,j))
    Count_Vectorizer = CountVectorizer(ngram_range=(i, j))  # change settings for unigram, bigram, trigram
    cachedStopwords = stopwords.words("english")
    porter = PorterStemmer()

# takes in a pandas dataframe and returns a feature vector and list of interests which act as labels
def getFeatureVectorAndLabels(dataframe):
    global Tfidf_Transformer, selector
    print()
    print("Converting dataframe to list of tweets and list of interests")
    listOfTweets, interestLabels = dfToTweetsAndInterests(dataframe)
    print("Stemming list of tweets")
    listOfTweets = stemList(porter, listOfTweets)  # change stemming algorithm here
    data_counts = Count_Vectorizer.fit_transform(listOfTweets)
    print("There are %s features" % data_counts.shape[1])
    print()
    temp_tfidf_transformer = TfidfTransformer(use_idf=True).fit(data_counts)
    Tfidf_Transformer = temp_tfidf_transformer
    tfidf_doc = TfidfTransformer(use_idf=True).fit_transform(data_counts)
    return tfidf_doc, interestLabels


# takes in a stemmer object defined in initalizeCV() and a list of strings to be stemmed
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


# Returns a cleaned dataset without punctuation and stopwords
def dfToTweetsAndInterests(dataframe):
    listOfTweets = []
    interestLabels = []
    for i in dataframe.index:
        tweet = dataframe["Tweet"][i]
        interest = dataframe["Label"][i]
        listOfTweets.append(tweet)
        interestLabels.append(interest)
    return listOfTweets, interestLabels


# train support vector machine classifier with given features and labels
def trainSVM(features, labels, kernelType):
    # starttime = datetime.datetime.now()
    if kernelType == 'sgd':
        clf = SGDClassifier().fit(features, labels)
    # kernelType can be linear, poly, rbf or sigmoid
    else:
        clf = SVC(kernel=kernelType).fit(features, labels)
    # print("Time taken to train SVMClassifier(" + kernelType + ") : " + str(datetime.datetime.now() - starttime))
    return clf


# returns an interest prediction
def predictInterest(targetHandle, classifier):
    data_counts = Count_Vectorizer.transform(targetHandle)
    tfidf_doc = TfidfTransformer(use_idf=True).fit(data_counts).transform(data_counts)
    prediction = classifier.predict(tfidf_doc)
    return prediction

# uses the cross_val_score method to calculate the accuracy of a model using kfold cross validation, with cv being the number of folds
def printKFoldScore(classifier, features, labels, name):
    kfold_score = cross_val_score(classifier, features, labels, cv=10)
    print("Accuracy for " + name + ": " + str(kfold_score.mean()))
    print()


# takes a prediction using the desired classifier and prints the classification report and confusion matrix
def printMetrics(classifier, features, labels, name):
    predictedList = classifier.predict(features)
    print()
    print("Classification report for " + name)
    print(classification_report(labels, predictedList))
    cm = confusion_matrix(labels, predictedList)
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
def testClassifier(classifier, dataframe):
    interestLabels, predictedInterests, listOfTweets = [], [], []
    print("Running on test set")
    for i in dataframe.index:
        tweet = dataframe["Tweet"][i]
        interest = dataframe["Label"][i]
        interestLabels.append(interest)
        predictedInterests.append(predictInterest([tweet], classifier))
        listOfTweets.append(tweet)
    if len(interestLabels) == len(predictedInterests):
        correct = 0
        print("False predictions:")
        for j in range(len(interestLabels)):
            if interestLabels[j] == predictedInterests[j]:
                correct += 1
            else:
                print(listOfTweets[j] + " is " + interestLabels[j] + " but predicted as " + str(predictedInterests[j][0]))
        print()
        print(str(correct) + "/" + str(len(interestLabels)) + " tweets predicted correctly")
        print()


def train(trainingSet):
    ngrams = [(1,1),(2,2),(3,3),(1,2),(1,3)]
    for (i,j) in ngrams:
        initializeCV(i,j)
        tfidf_doc, interestLabels = getFeatureVectorAndLabels(trainingSet)
        loadFromSave = False
        if loadFromSave:
            # joblib is an sklearn library that allows us to save / load the trained classifiers
            SVMClassifier = joblib.load('classifiers/svm.pkl')
            print("Classifier loaded from file")
        else:
            SVMClassifier = trainSVM(tfidf_doc, interestLabels, "sgd")
            joblib.dump(SVMClassifier, 'classifiers/svm.pkl')

        # kfold score for SVMClassifier
        printKFoldScore(SVMClassifier,tfidf_doc,interestLabels,"SVMClassifier")

        # classification report and confusion matrix for SVMClassifier
        printMetrics(SVMClassifier,tfidf_doc,interestLabels,"SVMClassifier")


    return SVMClassifier
