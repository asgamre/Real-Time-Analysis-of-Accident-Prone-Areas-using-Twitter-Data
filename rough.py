import re

string = "czechreality strip club porn anal accident backroom casting couch fantasy massage nubile priya rai bulgaria pic.twitter.com/6HEEQVOFQv"
tweet = re.sub(
    r'^(?:http|ftp)s?://' # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
    r'localhost|' # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|' # ...or ipv4
    r'\[?[A-F0-9]*:[A-F0-9:]+\]?)' # ...or ipv6
    r'(?::\d+)?' # optional port
    r'(?:/?|[/?]\S+)$', "", string)
print(tweet)
input("Press")
import pyap
import usaddress
print("1")
string = "Accident , left lane blocked Dallas 635 LBJ service Rd EB Greenville Ave DFWTraffic"
print(usaddress.parse(string))
print("2")
string = "Accident left lane blocked Dallas 635 LBJ service Rd EB Greenville Ave DFWTraffic".lower()
print(usaddress.parse(string))

addresses = pyap.parse(string, country='US')
for address in addresses:
        # shows found address
        print(address)
        # shows address parts
        print(address.as_dict())

def dfToTweetsAndInterests(dataframe):
    listOfTweets = []
    interestLabels = []
    for i in dataframe.index:
        tweet = dataframe["Tweet"][i]
        interest = dataframe["Label"][i]
        # if type(tweet) == str:
        listOfTweets.append(tweet)
        interestLabels.append(interest)
    return listOfTweets, interestLabels
