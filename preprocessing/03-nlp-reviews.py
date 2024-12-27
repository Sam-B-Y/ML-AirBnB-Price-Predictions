import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import time
from openai import OpenAI
from googletrans import Translator

# get api key from env
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key
)


sia = SentimentIntensityAnalyzer()
translator = Translator()

def compute_sentiment(text):
        sentiment = sia.polarity_scores(text)
        return sentiment['compound']

def compute_sentiment_with_nan(text):
        text = str(text)
        if not text.strip():
            return np.nan
        return compute_sentiment(text)


counter = 0

def process_reviews(df):
    def get_reviews_text(reviews):
        return ' '.join(str(reviews).split("---------------------------------"))

    # unused function that gets sentiment through nltk by first translating the reviews into english
    def get_translate_text(reviews):
        if pd.isnull(reviews) or not str(reviews).strip():
            return ''
        splitted = str(reviews).split("---------------------------------")
        for i in range(len(splitted)):
            text = str(splitted[i].strip())
            if len(text) < 5:
                splitted[i] = ''
                continue
            time.sleep(1)
            try:
                translation =  translator.translate(text, dest='en')
                if translation.src != 'en':
                    splitted[i] = translation.text
            except Exception as e:
                print(f"Error translating review segment: {e}")

        return ' '.join(splitted)
    
    # using chatgpt sentiment analysis instead of nltk
    def review_sentiment_chatgpt(reviews):
        global counter
        counter += 1
        print(f"Processing review {counter}")

        if pd.isnull(reviews) or not reviews.strip() or len(reviews) < 5:
            return np.nan
        review = reviews.strip()
        if not review:
            return np.nan
        
        review = review.replace("\n", " ") 
        review = review[:2000]
        messages = [
            {
                "role": "system",
                "content": "Classify each review's sentiment on a scale from 0 to 1, where 0 is very negative and 1 is very positive. The input may be in multiple languages. Output only a number."
            },
            {
                "role": "user",
                "content": f"Input:\n{review}\nOutput:"
            }
        ]
        try:
            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=1000
            )
            sentiment = response.choices[0].message.content.strip()
            sentiment = float(sentiment)
        except Exception as e:
            print(f"Error classifying sentiment: {e}")
            return np.nan
        if sentiment < 0.3:
            print(sentiment, review)
        return sentiment


    df['reviews_text'] = df['reviews'].apply(get_reviews_text)
    df['review_sentiment'] = df['reviews_text'].apply(review_sentiment_chatgpt)

    def average_review_length(reviews):
        reviews = str(reviews).split("---------------------------------")
        return np.mean([len(review.split()) for review in reviews]) if reviews else np.nan

    df['avg_review_length'] = df['reviews'].apply(average_review_length)

    df.drop(columns=['reviews', 'reviews_text'], inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df

def process_df(df):
    df = process_reviews(df)

    print("Reviews columns cleaned and encoded")

    return df


train_df = pd.read_csv('../data/train_nlp1.csv')
test_df = pd.read_csv('../data/test_nlp1.csv')

train_df = process_df(train_df)
test_df = process_df(test_df)

train_nlp2 = train_df[['review_sentiment', 'avg_review_length']]
train_nlp2.to_csv('../data/train_sentiment.csv', index=False)

test_nlp2 = test_df[['review_sentiment', 'avg_review_length']]

test_nlp2.to_csv('../data/test_sentiment.csv', index=False)