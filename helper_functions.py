from datasets import load_dataset
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm


def remove_rows_based_on_value(df, column_name, value):
    df = df[df[column_name] != value]
    return df.reset_index(drop=True)

def load_huggingface_dataset(dataset_name: str, train_split: float = 0.8, verbose: bool = False):
    np.random.seed(42)
    # load and train test split the dataset
    dataset = load_dataset(dataset_name)
    dataset = dataset['train'].train_test_split(0.2)
    train_df, test_df = dataset['train'].to_pandas(), dataset['test'].to_pandas()

    # remove rows with empty body
    train_df = remove_rows_based_on_value(train_df, 'body', '[removed]')
    train_df = remove_rows_based_on_value(train_df, 'body', '[deleted]')

    test_df = remove_rows_based_on_value(test_df, 'body', '[removed]')
    test_df = remove_rows_based_on_value(test_df, 'body', '[deleted]')

    # verbose for verification
    if verbose:
        print(f"Dataset loaded: {dataset_name}")
        print(f"Train size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")

    return train_df, test_df

def preprocess_dataframe_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f'Column {column_name} does not exist in the DataFrame.')

    # Initialize the lemmatizer and stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Define a function to preprocess individual text entries
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove stopwords
        text = ' '.join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    # Apply the preprocessing to the specified column
    print(f'Preprocessing text in the {column_name} column...')
    df[column_name] = df[column_name].apply(preprocess_text)
    return df