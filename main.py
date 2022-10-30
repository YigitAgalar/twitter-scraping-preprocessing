
import snscrape.modules.twitter as sntwitter
import pandas as pd

query= '(#TSLA) lang:en until:2022-10-01 since:2017-01-01'

tweets=[]
limit=20000



for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    if len(tweets)==limit:
        break
    else:
        tweets.append([tweet.date,tweet.user.username,tweet.content])



df= pd.DataFrame(tweets, columns=['Date','User','Tweet'])
df.to_csv('TSLAtweets.csv')