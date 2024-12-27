import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import json
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import nltk
import re
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import re
from fuzzywuzzy import process

sia = SentimentIntensityAnalyzer()

def compute_sentiment(text):
        # compute sentiment using nltk's sentiment intensity analyzer
        sentiment = sia.polarity_scores(text)
        return sentiment['compound']

def compute_sentiment_with_nan(text):
        text = str(text)
        if not text.strip():
            return np.nan
        return compute_sentiment(text)

def clean_text(text):
    text = str(text).lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def process_amenities(df):
    def normalize_amenity(amenity):
        # remove punctuation and convert to lowercase
        amenity = amenity.lower()
        amenity = re.sub(r'[^\w\s]', '', amenity)
        return amenity.strip()

    def extract_amenities(amenities_str):
        try:
            return json.loads(amenities_str)
        except json.JSONDecodeError:
            return []

    # Apply normalization
    df['normalized_amenities'] = df['amenities'].apply(lambda x: [normalize_amenity(amenity) for amenity in extract_amenities(x)])

    # Create a master list of standard amenities
    standard_amenities = [
        'wifi', 'tv', 'air conditioning', 'kitchen',
        'parking', 'pool', 'gym', 'washer', 'dryer', 'breakfast', 'view', 
    ]

    def map_amenities(amenity_list):
        # find if the current amenity is similar to any of the standard amenities, if so add it to the mapped amenities
        mapped_amenities = set()
        for amenity in amenity_list:
            match, score = process.extractOne(amenity, standard_amenities)
            if score >= 80: # Minimum score for a match
                mapped_amenities.add(match)
        return list(mapped_amenities)

    df['mapped_amenities'] = df['normalized_amenities'].apply(map_amenities)

    mlb = MultiLabelBinarizer()
    # encode the mapped amenities into binray columns
    amenities_encoded = mlb.fit_transform(df['mapped_amenities'])
    amenities_df = pd.DataFrame(amenities_encoded, columns=mlb.classes_)

    # Combine with the original DataFrame
    df.reset_index(drop=True, inplace=True)

    df = pd.concat([df, amenities_df], axis=1)
    df.drop(columns=['amenities', 'normalized_amenities', 'mapped_amenities'], inplace=True)

    return df

def process_df(df):
    # convert boolean columns to binary
    df['host_is_superhost'] = df['host_is_superhost'].replace({True: 1, False: 0})
    df['host_has_profile_pic'] = df['host_has_profile_pic'].replace({True: 1, False: 0})
    df['host_identity_verified'] = df['host_identity_verified'].replace({True: 1, False: 0})
    df['has_availability'] = df['has_availability'].replace({True: 1, False: 0})
    df['instant_bookable'] = df['instant_bookable'].replace({True: 1, False: 0})

    print("Converted boolean columns to integers")

    # convert dates into relative time since a reference date
    current_date = pd.to_datetime('October 23, 2024')

    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['host_tenure_days'] = (current_date - df['host_since']).dt.days

    df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
    df['time_since_first_review'] = (current_date - df['first_review']).dt.days

    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['time_since_last_review'] = (current_date - df['last_review']).dt.days

    df.drop(columns=['host_since', 'first_review', 'last_review'], inplace=True)

    print("Converted date columns to datetime since the reference date")

    # one hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    room_type_encoded = encoder.fit_transform(df[['room_type']])

    room_type_encoded_df = pd.DataFrame(
        room_type_encoded, 
        columns=encoder.get_feature_names_out(['room_type']),
        index=df.index
    )

    df = pd.concat([df, room_type_encoded_df], axis=1)
    df.drop(columns=['room_type', 'property_type'], inplace=True)

    print("Room type and property type columns encoded")

    # score the host response time from text to numerical
    response_time_mapping = {
        'within an hour': 1,
        'within a few hours': 2,
        'within a day': 4,
        'a few days or more': 10,
        np.nan: np.nan
    }

    df['host_response_time'] = df['host_response_time'].map(response_time_mapping)

    df['host_verifications'] = df['host_verifications'].apply(ast.literal_eval)
    verification_weights = {
        'phone': 2, 
        'email': 1,
        'work_email': 0 
    }


    def calculate_verification_score(verifications):
        score = sum(verification_weights.get(item, 0) for item in verifications)
        return score

    # convert verification methods to a numerical score
    df['host_verification_score'] = df['host_verifications'].apply(calculate_verification_score)
    df.drop(columns=['host_verifications'], inplace=True)


    print("Host response time and verifications columns cleaned")

    # one hot encode neighbourhood columns (will be removed later, but temporary solution)
    neighbourhood_group_encoded = encoder.fit_transform(df[['neighbourhood_group_cleansed']])
    neighbourhood_group_encoded_df = pd.DataFrame(
        neighbourhood_group_encoded, 
        columns=encoder.get_feature_names_out(['neighbourhood_group_cleansed']),
        index=df.index
    )

    df = pd.concat([df, neighbourhood_group_encoded_df], axis=1)
    df.drop(columns=['neighbourhood_group_cleansed'], inplace=True)

    neighbourhood_cleansed_encoded = encoder.fit_transform(df[['neighbourhood_cleansed']])
    neighbourhood_cleansed_encoded_df = pd.DataFrame(
        neighbourhood_cleansed_encoded, 
        columns=encoder.get_feature_names_out(['neighbourhood_cleansed']),
        index=df.index
    )

    df = pd.concat([df, neighbourhood_cleansed_encoded_df], axis=1)
    df.drop(columns=['neighbourhood_cleansed'], inplace=True)

    print("Neighbourhood group cleansed column encoded")

    # check if the bathrooms are shared
    df['bathrooms_shared'] = df['bathrooms_text'].apply(lambda x: np.nan if pd.isnull(x) else 1 if 'shared' in str(x) else 0)
    df.drop(columns=['bathrooms_text'], inplace=True)

    print("Neighbourhood cleansed column encoded")

    df = process_amenities(df)

    print("Amenities column cleaned and encoded")

    return df

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

print(train_df.shape, test_df.shape)

processed_train_df = process_df(train_df)
processed_test_df = process_df(test_df)

processed_train_df.to_csv('../data/train_cleaned.csv', index=False)
processed_test_df.to_csv('../data/test_cleaned.csv', index=False)