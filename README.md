# Twitter-Sentiment-Analysis-Using-Streamlit
The Streamlit framework makes deploying ML models just as easy as a few function calls.

## Sentiment Classification
Sentiment classification is a classic problem in NLP regarding understanding whether a sentence is positive or negative.
For example, “I love Python!” should be classified as a positive sentiment while “Python is the worst!” should be classified as a negative sentiment.

### Flair
I used Flair, a library for state-of-the-art NLP classification.

### Data
The dataset used for this is Sentiment140. The Sentiment140 dataset is a collection of 1.6 million tweets labeled as 0, negative sentiment, or 4, positive sentiment.
The dataset can be downloaded from [here](https://www.kaggle.com/kazanova/sentiment140#training.1600000.processed.noemoticon.csv)
### Twitter Scraper
The idea was to scrape Twitter for recent tweets with a given query, classify each one into a positive/negative sentiment, and calculate the ratio of positive to negative tweets using twitterscraper

In main.py file after few imports I built the loading classification model along with the title of the page
after importing preprocessing and as long as the input not being empty we 
1. Pre-process the tweet
2. Make predictions (with a spinner as we wait)
3. And show the predictions

<p align="center">
  <img width="875" height="539" src="https://miro.medium.com/max/875/1*XV4dAKO62pf9haLUpAF_-Q.png">
</p>

After searching for a topic on Twitter and finding the positive to negative tweet ratio, after some tuning and testing the full streamlit app looks like

<p align="center">
  <img width="496" height="722" src="https://miro.medium.com/max/750/1*esHRI9XA0gwwug8lO5QSgw.gif">
</p>
