"""
Data utility functions for fake news detection project.
Includes functions for loading data, computing statistics, and visualization helpers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(file_path):
    """
    Load the dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(file_path)
    return df


def get_basic_stats(df):
    """
    Get basic statistics about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing basic statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    return stats


def analyze_class_distribution(df, label_col='real'):
    """
    Analyze the distribution of classes (real vs fake).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    label_col : str
        Name of the label column
        
    Returns:
    --------
    dict
        Dictionary containing class distribution statistics
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataframe")
    
    class_counts = df[label_col].value_counts()
    class_props = df[label_col].value_counts(normalize=True) * 100
    
    stats = {
        'counts': class_counts.to_dict(),
        'proportions': class_props.to_dict(),
        'total': len(df),
        'balance_ratio': min(class_counts) / max(class_counts)
    }
    
    return stats


def plot_class_distribution(df, label_col='real', save_path=None):
    """
    Plot the distribution of classes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    label_col : str
        Name of the label column
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    sns.countplot(data=df, x=label_col, ax=axes[0], palette='viridis')
    axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Label (0=Fake, 1=Real)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xticklabels(['Fake', 'Real'])
    
    # Add count labels on bars
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%d', fontsize=10)
    
    # Pie chart
    class_counts = df[label_col].value_counts()
    axes[1].pie(class_counts.values, labels=['Fake', 'Real'], autopct='%1.1f%%',
                colors=['#ff6b6b', '#4ecdc4'], startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_text_length(df, text_col='title', label_col='real'):
    """
    Analyze text length statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    text_col : str
        Name of the text column
    label_col : str
        Name of the label column
        
    Returns:
    --------
    pd.DataFrame
        Statistics by class
    """
    df = df.copy()
    df['text_length'] = df[text_col].astype(str).str.len()
    df['word_count'] = df[text_col].astype(str).str.split().str.len()
    
    stats = df.groupby(label_col).agg({
        'text_length': ['mean', 'median', 'std', 'min', 'max'],
        'word_count': ['mean', 'median', 'std', 'min', 'max']
    }).round(2)
    
    return stats


def plot_text_length_distribution(df, text_col='title', label_col='real', save_path=None):
    """
    Plot text length distribution by class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    text_col : str
        Name of the text column
    label_col : str
        Name of the label column
    save_path : str, optional
        Path to save the plot
    """
    df = df.copy()
    df['text_length'] = df[text_col].astype(str).str.len()
    df['word_count'] = df[text_col].astype(str).str.split().str.len()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Character length distribution
    for label in [0, 1]:
        label_name = 'Fake' if label == 0 else 'Real'
        data = df[df[label_col] == label]['text_length']
        axes[0, 0].hist(data, alpha=0.6, label=label_name, bins=50)
    axes[0, 0].set_xlabel('Character Length', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Character Length Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Word count distribution
    for label in [0, 1]:
        label_name = 'Fake' if label == 0 else 'Real'
        data = df[df[label_col] == label]['word_count']
        axes[0, 1].hist(data, alpha=0.6, label=label_name, bins=50)
    axes[0, 1].set_xlabel('Word Count', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot for character length
    df_melted = df.melt(id_vars=[label_col], value_vars=['text_length'], 
                        var_name='metric', value_name='value')
    sns.boxplot(data=df, x=label_col, y='text_length', ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_xlabel('Label (0=Fake, 1=Real)', fontsize=12)
    axes[1, 0].set_ylabel('Character Length', fontsize=12)
    axes[1, 0].set_title('Character Length by Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticklabels(['Fake', 'Real'])
    
    # Box plot for word count
    sns.boxplot(data=df, x=label_col, y='word_count', ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_xlabel('Label (0=Fake, 1=Real)', fontsize=12)
    axes[1, 1].set_ylabel('Word Count', fontsize=12)
    axes[1, 1].set_title('Word Count by Class', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticklabels(['Fake', 'Real'])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_source_domains(df, source_col='source_domain', label_col='real', top_n=20):
    """
    Analyze source domain distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    source_col : str
        Name of the source domain column
    label_col : str
        Name of the label column
    top_n : int
        Number of top domains to return
        
    Returns:
    --------
    pd.DataFrame
        Top domains with their counts and class distribution
    """
    domain_stats = df.groupby([source_col, label_col]).size().unstack(fill_value=0)
    domain_stats['total'] = domain_stats.sum(axis=1)
    domain_stats = domain_stats.sort_values('total', ascending=False).head(top_n)
    domain_stats.columns = ['Fake', 'Real', 'Total']
    
    return domain_stats


def plot_source_domains(df, source_col='source_domain', label_col='real', top_n=15, save_path=None):
    """
    Plot top source domains.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    source_col : str
        Name of the source domain column
    label_col : str
        Name of the label column
    top_n : int
        Number of top domains to plot
    save_path : str, optional
        Path to save the plot
    """
    domain_counts = df[source_col].value_counts().head(top_n)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top domains bar plot
    axes[0].barh(range(len(domain_counts)), domain_counts.values, color='steelblue')
    axes[0].set_yticks(range(len(domain_counts)))
    axes[0].set_yticklabels(domain_counts.index, fontsize=10)
    axes[0].set_xlabel('Number of Articles', fontsize=12)
    axes[0].set_title(f'Top {top_n} Source Domains', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Domain distribution by class
    top_domains = domain_counts.index.tolist()
    df_top = df[df[source_col].isin(top_domains)]
    domain_class = pd.crosstab(df_top[source_col], df_top[label_col])
    domain_class = domain_class.reindex(top_domains)
    domain_class.plot(kind='barh', stacked=True, ax=axes[1], 
                      color=['#ff6b6b', '#4ecdc4'], width=0.8)
    axes[1].set_xlabel('Number of Articles', fontsize=12)
    axes[1].set_title(f'Top {top_n} Domains by Class', fontsize=14, fontweight='bold')
    axes[1].legend(['Fake', 'Real'], fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_tweet_distribution(df, tweet_col='tweet_num', label_col='real'):
    """
    Analyze tweet number distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    tweet_col : str
        Name of the tweet number column
    label_col : str
        Name of the label column
        
    Returns:
    --------
    pd.DataFrame
        Statistics by class
    """
    stats = df.groupby(label_col)[tweet_col].agg([
        'mean', 'median', 'std', 'min', 'max', 'sum'
    ]).round(2)
    
    return stats


def plot_tweet_distribution(df, tweet_col='tweet_num', label_col='real', save_path=None):
    """
    Plot tweet number distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    tweet_col : str
        Name of the tweet number column
    label_col : str
        Name of the label column
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    for label in [0, 1]:
        label_name = 'Fake' if label == 0 else 'Real'
        data = df[df[label_col] == label][tweet_col]
        axes[0, 0].hist(data, alpha=0.6, label=label_name, bins=50)
    axes[0, 0].set_xlabel('Number of Tweets', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Tweet Number Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(data=df, x=label_col, y=tweet_col, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_xlabel('Label (0=Fake, 1=Real)', fontsize=12)
    axes[0, 1].set_ylabel('Number of Tweets', fontsize=12)
    axes[0, 1].set_title('Tweet Number by Class', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticklabels(['Fake', 'Real'])
    
    # Log scale histogram (for better visualization if there are outliers)
    for label in [0, 1]:
        label_name = 'Fake' if label == 0 else 'Real'
        data = df[df[label_col] == label][tweet_col] + 1  # +1 to avoid log(0)
        axes[1, 0].hist(np.log10(data), alpha=0.6, label=label_name, bins=50)
    axes[1, 0].set_xlabel('Log10(Number of Tweets + 1)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Tweet Number Distribution (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Violin plot
    sns.violinplot(data=df, x=label_col, y=tweet_col, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_xlabel('Label (0=Fake, 1=Real)', fontsize=12)
    axes[1, 1].set_ylabel('Number of Tweets', fontsize=12)
    axes[1, 1].set_title('Tweet Number Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticklabels(['Fake', 'Real'])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_wordcloud(df, text_col='title', label_col='real', label_value=1, save_path=None):
    """
    Generate word cloud for a specific class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    text_col : str
        Name of the text column
    label_col : str
        Name of the label column
    label_value : int
        Label value (0 for fake, 1 for real)
    save_path : str, optional
        Path to save the word cloud
    """
    label_name = 'Real' if label_value == 1 else 'Fake'
    text = ' '.join(df[df[label_col] == label_value][text_col].astype(str).tolist())
    
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         max_words=100, colormap='viridis').generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {label_name} News', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_report(df, label_col='real'):
    """
    Print a comprehensive summary report of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    label_col : str
        Name of the label column
    """
    print("=" * 80)
    print("DATASET SUMMARY REPORT")
    print("=" * 80)
    print(f"\nTotal Records: {len(df):,}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "-" * 80)
    print("CLASS DISTRIBUTION")
    print("-" * 80)
    class_dist = analyze_class_distribution(df, label_col)
    print(f"Real News (1): {class_dist['counts'].get(1, 0):,} ({class_dist['proportions'].get(1, 0):.2f}%)")
    print(f"Fake News (0): {class_dist['counts'].get(0, 0):,} ({class_dist['proportions'].get(0, 0):.2f}%)")
    print(f"Balance Ratio: {class_dist['balance_ratio']:.3f} (1.0 = perfectly balanced)")
    
    print("\n" + "-" * 80)
    print("MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in df.columns:
        if missing[col] > 0:
            print(f"{col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
        else:
            print(f"{col}: No missing values")
    
    print("\n" + "-" * 80)
    print("DATA TYPES")
    print("-" * 80)
    for col, dtype in df.dtypes.items():
        print(f"{col}: {dtype}")
    
    print("\n" + "=" * 80)

