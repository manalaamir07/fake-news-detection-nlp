"""
Preprocessing functions for fake news detection project.
Includes text cleaning, feature engineering, and data preparation utilities.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def clean_text(text, remove_urls=True, remove_special_chars=True, lowercase=True):
    """
    Clean text by removing URLs, special characters, and converting to lowercase.
    
    Parameters:
    -----------
    text : str
        Input text to clean
    remove_urls : bool
        Whether to remove URLs
    remove_special_chars : bool
        Whether to remove special characters (keeps alphanumeric and spaces)
    lowercase : bool
        Whether to convert to lowercase
        
    Returns:
    --------
    str
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    if remove_urls:
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
    
    if remove_special_chars:
        # Keep only alphanumeric characters and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    if lowercase:
        text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_text_features(df, text_col='title'):
    """
    Extract features from text column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    text_col : str
        Name of the text column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional feature columns
    """
    df = df.copy()
    
    # Character length
    df['char_length'] = df[text_col].astype(str).str.len()
    
    # Word count
    df['word_count'] = df[text_col].astype(str).str.split().str.len()
    
    # Average word length
    df['avg_word_length'] = df['char_length'] / (df['word_count'] + 1)  # +1 to avoid division by zero
    
    # Number of uppercase letters
    df['uppercase_count'] = df[text_col].astype(str).str.findall(r'[A-Z]').str.len()
    
    # Number of digits
    df['digit_count'] = df[text_col].astype(str).str.findall(r'\d').str.len()
    
    # Number of exclamation marks (escape special regex characters)
    df['exclamation_count'] = df[text_col].astype(str).str.count(r'!')
    
    # Number of question marks (escape special regex characters)
    df['question_count'] = df[text_col].astype(str).str.count(r'\?')
    
    return df


def encode_source_domain(df, source_col='source_domain', method='label'):
    """
    Encode source domain column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    source_col : str
        Name of the source domain column
    method : str
        Encoding method: 'label' for label encoding, 'frequency' for frequency encoding
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded source domain
    """
    df = df.copy()
    
    if method == 'label':
        le = LabelEncoder()
        df['source_domain_encoded'] = le.fit_transform(df[source_col].astype(str))
    elif method == 'frequency':
        # Frequency encoding
        domain_counts = df[source_col].value_counts().to_dict()
        df['source_domain_encoded'] = df[source_col].map(domain_counts)
    else:
        raise ValueError("Method must be 'label' or 'frequency'")
    
    return df


def normalize_tweet_num(df, tweet_col='tweet_num', method='log'):
    """
    Normalize tweet number column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    tweet_col : str
        Name of the tweet number column
    method : str
        Normalization method: 'log' for log transformation, 'minmax' for min-max scaling
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with normalized tweet number
    """
    df = df.copy()
    
    if method == 'log':
        # Log transformation (add 1 to avoid log(0))
        df['tweet_num_normalized'] = np.log1p(df[tweet_col].fillna(0))
    elif method == 'minmax':
        # Min-max scaling
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df['tweet_num_normalized'] = scaler.fit_transform(df[[tweet_col]].fillna(0))
    else:
        raise ValueError("Method must be 'log' or 'minmax'")
    
    return df


def handle_missing_values(df, strategy='fill'):
    """
    Handle missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy to handle missing values: 'fill', 'drop', or 'keep'
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with handled missing values
    """
    df = df.copy()
    
    if strategy == 'fill':
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(0)
    elif strategy == 'drop':
        # Drop rows with missing values
        df = df.dropna()
    elif strategy == 'keep':
        # Keep missing values as is
        pass
    else:
        raise ValueError("Strategy must be 'fill', 'drop', or 'keep'")
    
    return df


def prepare_data(df, text_col='title', label_col='real', 
                 clean_text_flag=True, extract_features_flag=True,
                 encode_source_flag=True, normalize_tweet_flag=True,
                 handle_missing_flag=True, missing_strategy='fill'):
    """
    Complete data preparation pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    text_col : str
        Name of the text column
    label_col : str
        Name of the label column
    clean_text_flag : bool
        Whether to clean text
    extract_features_flag : bool
        Whether to extract text features
    encode_source_flag : bool
        Whether to encode source domain
    normalize_tweet_flag : bool
        Whether to normalize tweet numbers
    handle_missing_flag : bool
        Whether to handle missing values
    missing_strategy : str
        Strategy for handling missing values
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    df = df.copy()
    
    # Handle missing values
    if handle_missing_flag:
        df = handle_missing_values(df, strategy=missing_strategy)
    
    # Clean text
    if clean_text_flag:
        df[text_col + '_cleaned'] = df[text_col].apply(clean_text)
        # Use cleaned text for feature extraction
        text_col_used = text_col + '_cleaned'
    else:
        text_col_used = text_col
    
    # Extract text features
    if extract_features_flag:
        df = extract_text_features(df, text_col=text_col_used)
    
    # Encode source domain
    if encode_source_flag:
        df = encode_source_domain(df, method='label')
    
    # Normalize tweet numbers
    if normalize_tweet_flag:
        df = normalize_tweet_num(df, method='log')
    
    return df


def split_data(df, label_col='real', test_size=0.2, random_state=42, stratify=True):
    """
    Split data into train and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    label_col : str
        Name of the label column
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to stratify the split based on labels
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) dataframes
    """
    if stratify:
        stratify_col = df[label_col]
    else:
        stratify_col = None
    
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
    )
    
    return X_train, X_test, y_train, y_test


def save_processed_data(df, file_path):
    """
    Save processed dataframe to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    file_path : str
        Path to save the CSV file
    """
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")


def load_processed_data(file_path):
    """
    Load processed dataframe from CSV.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_csv(file_path)
    return df

