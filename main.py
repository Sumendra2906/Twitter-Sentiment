import datetime as dt
import re
import string
import tweepy
import pandas as pd
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier

# Set page title
st.title('Twitter Sentiment Analysis')

# Load classification model
with st.spinner('Loading classification model...'):
    classifier = TextClassifier.load('model-saves/best-model.pt')

# Preprocess function
allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'

# data preprocessing


def remove_punc(text):
    nopunc = [char for char in text if char not in string.punctuation]
    return ''.join(nopunc)


def clean2(text):
    text = [char for char in text if char in allowed_chars]
    return ''.join(text)


def clean(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # removes the @ mentions
    text = re.sub(r'#', '', text)  # removes the '#'
    text = re.sub(r'https?:\/\/\S+', '', text)  # removes hyperlinks
    text = remove_punc(text)
    text = clean2(text)
    return text


label_dict = {'0': 'Negative', '4': 'Positive'}


### SINGLE TWEET CLASSIFICATION ###
st.subheader('Single tweet classification')

# Get sentence input, clean it, and convert to flair.data.Sentence format
tweet_input = st.text_input('Tweet:')

if tweet_input != '':
    # Pre-process tweet
    sentence = Sentence(clean(tweet_input))

    # Make predictions
    with st.spinner('Predicting...'):
        classifier.predict(sentence)

    # Show predictions

    if len(sentence.labels) > 0:
        st.write('Prediction:')
        st.write(label_dict[sentence.labels[0].value] + ' with ',
                 sentence.labels[0].score*100, '% confidence')


# Authenticating with the twitter api to pull tweets
#log = pd.read_csv('Login.csv')

consumerKey = st.secrets["API"]
consumerSecret = st.secrets["API_SECRET"]

accessToken = st.secrets["ACCESS_TOKEN"]
accessSecret = st.secrets["ACCESS_TOKEN_SECRET"]

authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
authenticate.set_access_token(accessToken, accessSecret)

# Create the API object while passing in the auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)

### TWEET SEARCH AND CLASSIFY ###
st.subheader('Search Twitter for Query')

# Get user input
query = st.text_input('Query:', '#')
frm_date = st.text_input('From Date (YYYY-MM-DD)')
n = st.text_input('Number of Tweets')

# As long as the query is valid (not empty or equal to '#')...
if st.button("Search and Analyse"):
    if query != '' and query != '#':
        st.success(f'Searching for and analyzing {query}...')
        # Get English tweets from the past 2 weeks
        tweets = tweepy.Cursor(api.search, q=query, lang="en",
                               since=frm_date, tweet_mode='extended').items(int(n))

        # Initialize empty dataframe
        tweet_data = pd.DataFrame({
            'tweet': [],
            'predicted-sentiment': [],
            'confidence': []
        })

        # Keep track of positive vs. negative tweets
        pos_vs_neg = {'0': 0, '4': 0}

        list_tweets = [tweet for tweet in tweets]
        # Add data for each tweet
        for tweet in list_tweets:

            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:
                text = tweet.full_text
            # Skip iteration if tweet is empty
            if text in ('', ' '):
                continue
            # Make predictions
            sentence = Sentence(clean(text))
            classifier.predict(sentence)
            sentiment = sentence.labels[0]
            # Keep track of positive vs. negative tweets
            pos_vs_neg[sentiment.value] += 1
            # Append new data
            tweet_data = tweet_data.append(
                {'tweet': text, 'predicted-sentiment': label_dict[sentiment.value], 'confidence': sentiment.score}, ignore_index=True)

# Show query data and sentiment if available
try:
    st.write(tweet_data)
    try:
        st.write('Positive to negative tweet ratio:',
                 pos_vs_neg['4']/pos_vs_neg['0'])
    except ZeroDivisionError:  # if no negative tweets
        st.write('All postive tweets')
except NameError:  # if no queries have been made yet
    pass
