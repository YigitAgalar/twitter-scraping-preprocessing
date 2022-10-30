#data manipulation
import pandas as pd
import numpy as np
import re
import string

#emoji handling
import demoji

#text processing
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
from textblob import TextBlob

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


stop_words = set(stopwords.words('english'))
df = pd.read_csv('TSLAtweets.csv')
#df.head()
df.rename(columns={'Unnamed: 0':'index'},inplace=True)
df.set_index('index',inplace=True)
df['wordcount_bc']=df['Tweet'].map(lambda x:len(x.split()))



def CleanTXT(text):
    text = text.lower() # text lowered
    
    text = demoji.replace(text,"") # emojis removed
    
    text = re.sub(r'\n',' ',text) # remove \n
    
    text= text.translate(str.maketrans("","",string.punctuation)) #punctuation removed
    
    text = re.sub(r'@[A-Za-z0-9]','',text) # @mentions removed
    
    text = re.sub(r'\@\w+|\#','',text) # remove hashtags and @ mentions 

    #https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    text = re.sub(r'https?:\/\/\S+','',text,flags=re.MULTILINE) # remove link 
    
    #filtering stopwords
    filtered_words = [word for word in TextBlob(text).words if word not in stop_words]
    
    #lemmatization
    lemmatized_words = [word.lemmatize('v') for word in filtered_words]
    
    #stemming
    #stemmed_words = [word.stem() for word in lemmatized_words]
    
    return " ".join(lemmatized_words)


df['cleanTweet'] = df['Tweet'].map(CleanTXT) 

df['wordcount_ac']=df['cleanTweet'].map(lambda x:len(x.split()))


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df['Subjectivity']= df['cleanTweet'].map(getSubjectivity)
df['Polarity']=df['cleanTweet'].map(getPolarity)



allWords = ' '.join([twts for twts in df['cleanTweet']])
wordCloud = WordCloud(width= 1000,height=500,random_state=35,max_font_size=110,collocations=False).generate(allWords)

figure = plt.figure(figsize=(15,7))
plt.imshow(wordCloud,interpolation = 'bilinear')
plt.axis('off')
plt.show()