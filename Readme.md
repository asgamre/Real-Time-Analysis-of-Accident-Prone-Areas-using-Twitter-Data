This is the term project for the class CS 6320 Natural Language Processing.

Connector.py is the entry point to the project.

Credentials.py will have credentials supplied by Twitter's development account. 
Connector.py has an "init" method which calls the train and test methods in Classifier.py.
Classifier.py has all the logic for Stemming, Extracting features, training the multinomial NB classifier on the training set and testing the classifier on the test set.
Preprocess.py has the preprocessing steps which will remove the non-ascii characters in a tweet, which includes Emoticons, URLs, etc. It also removes stopwords.


Run Connector.py as
>> python Connector.py

On the command line, we will see the performance and classification report of the classifier over different n-gram ranges. We finally see which n-gram range gives the best performance and evaluate the test set on that range.
At the end of train and test, there is a section for live tweets where we can tweets made from near the Dallas-Fort Worth area containing accident related keywords would appear on the console and along with those tweets would appear the prediction for that tweet whether it is situational or non-situational.