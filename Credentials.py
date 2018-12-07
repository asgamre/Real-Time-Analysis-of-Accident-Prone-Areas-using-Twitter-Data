from tweepy import OAuthHandler

def authenticate():
    ckey="Insert Key From Twitter Dev Account Here"
    csecret="Insert Key From Twitter Dev Account Here"
    atoken="Insert Key From Twitter Dev Account Here"
    asecret="Insert Key From Twitter Dev Account Here"

    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    return auth