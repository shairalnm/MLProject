import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("C:/Users/saira/Twitter Sentiment Analysis/trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None, names=cols, encoding='iso-8859-1')
df.head(5)
df.dtypes

df['sentiment'].value_counts()

df.drop(['id','date','query_string','user'],axis=1,inplace=True)

df[df['sentiment']==0].head(5)

df[df['sentiment']==4].head(5)

df[df['sentiment']==4].index

df[df['sentiment']==0].index

df['sentiment'] = df['sentiment'].map({0:0, 4:1})

df['text'].head(75)

df['pre_clean_len'] = [len(t) for t in df.text]
from pprint import pprint
data_dict = {
    'sentiment':{
        'type':df.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df.text.dtype,
        'description':'tweet text'
    },
    'pre_clean_len':{
        'type':df.pre_clean_len.dtype,
        'description':'Length of the tweet before cleaning'
    },
    'dataset_shape':df.shape
}
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()

df['text'].head(20)

df[df.pre_clean_len > 140].head(25)

df.dtypes

# Data Cleansing

from html2text import unescape
df.text = df.text.apply(unescape, unicode_snob=True)

df['text'] = df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

df['text'] = df['text'].replace(r'@[A-Za-z0-9]+', '', regex=True)



df.head(50)





df.head(50)

df['text'] = df['text'].map(lambda x : x.lower())

df['text'].head(5)

df['text'] = df['text'].replace(r'[^\w\s]', '', regex=True)

df['text'].head(5)

df.head(5)

df.drop(['pre_clean_len'], axis=1, inplace= True)

df.head(5)
df.to_csv('cleantweet.csv')

df.shape
