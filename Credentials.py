from tweepy import OAuthHandler

def authenticate():
    ckey="XflayFhqiT3CpTTUy1HlAYKgT"
    csecret="nCQnXwgNFR1hCLJ0woc6mDW94Z1O2X9AwOY0yerwrKcHEhj6Zo"
    atoken="116456043-BhynBGhHhNhOvGPs1LhzRNYZSsVHo1LhlM6Lzj2y"
    asecret="NZBtZbd2isPfSqkB4nFNteJKkFzigW5tZExMoDk6cRaRt"

    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    return auth