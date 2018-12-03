import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))

def preprocess(tweet):
    tweet = str(tweet)
    tweet = re.sub(r'(https?:\/\/)(\s)?(www\.)?(\s?)(\w+\.)*([\w\-\s]+\/)*([\w-]+)\/?', '', tweet)
    tweet = ''.join([x for x in tweet if ord(x) < 128])
    exclude = set('!"$%&\',.()*+-/:;<=>?@[\]^_`{|}~#')
    tweet = ''.join(ch for ch in tweet if ch not in exclude)
    words = word_tokenize(tweet)
    filtered_tweet = [w for w in words if w not in stop_words]
    return ' '.join(filtered_tweet)