import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import re
# from googletrans import Translator
# translator = Translator()

sia = SentimentIntensityAnalyzer()

def compute_sentiment(text):
        sentiment = sia.polarity_scores(text)
        return sentiment['compound']

def compute_sentiment_with_nan(text):
        text = str(text)
        if not text.strip():
            return np.nan
        return compute_sentiment(text)

def process_name(df):
    df['combined_text'] = df['name'] + ' ' + df['description']

    # getting the sentiment of the name and description of a listing (assuming it is in English)
    df['combined_text_sentiment'] = df['combined_text'].apply(compute_sentiment_with_nan)
    df['listing_length'] = df['combined_text'].str.len()
    df['listing_length'] = df['listing_length'].fillna(0).astype(int)
    df.drop(columns=['name', 'description', 'combined_text'], inplace=True)

    return df

def process_df(df):
    df = process_name(df)

    print("Name and description columns cleaned and encoded")

    return df


train_df = pd.read_csv('../data/train_cleaned.csv')
test_df = pd.read_csv('../data/test_cleaned.csv')

print(train_df.shape, test_df.shape)

train_df = process_df(train_df)
test_df = process_df(test_df)

train_df.to_csv('../data/train_nlp1.csv', index=False)
test_df.to_csv('../data/test_nlp1.csv', index=False)