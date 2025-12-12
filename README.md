# Fake News Detection - NLP Project

A machine learning project for detecting fake news using Natural Language Processing techniques. This project implements multiple modeling approaches including classical machine learning, deep learning, and transformer-based models to classify news articles as real or fake.

## Project Overview

This project aims to classify news articles as either real or fake using various NLP and machine learning techniques. The dataset used is from FakeNewsNet, containing news articles with their titles, source domains, tweet engagement metrics, and labels. The project follows a comprehensive workflow from data exploration through model implementation and comparison.

## Dataset

The dataset contains news articles with the following features:
- title: Title of the article
- news_url: URL of the article
- source_domain: Web domain where article was posted
- tweet_num: Number of retweets for this article
- real: Label column (1 = real, 0 = fake)

Dataset Source: FakeNewsNet (cleaned and combined)

Dataset Size: 23,196 articles (17,441 real, 5,755 fake)

## Project Structure

fake-news-detection/
├── data/
│   ├── raw/                 # Original dataset (not tracked in git)
│   └── processed/           # Cleaned CSVs used by notebooks
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_models.ipynb
│   ├── 04_deep_learning.ipynb
│   └── 05_transformers.ipynb
├── results/
│   ├── visualizations/      # Saved plots and charts
│   └── metrics/             # Model performance metrics (CSV files)
├── src/
│   ├── data_utils.py
│   ├── preprocessing.py
│   └── models.py
├── requirements.txt
├── environment.yml         # Optional conda environment
├── README.md
└── .gitignore


## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fake-news-detection
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Make sure virtual environment is activated first
pip install -r requirements.txt
```

Note: If pip command is not found, use python3 -m pip install -r requirements.txt or pip3 install -r requirements.txt

Or using conda:
```bash
conda env create -f environment.yml
conda activate fake-news-detection
```

4. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)
This notebook performs comprehensive exploratory data analysis on the FakeNewsNet dataset. It includes:
- Dataset structure and basic statistics
- Class distribution analysis (real vs fake news)
- Text statistics and visualizations (title length, word count)
- Source domain patterns and analysis
- Tweet engagement metrics analysis
- Missing values identification
- Word cloud visualizations
- Summary of key findings and insights

### 2. Preprocessing (`02_preprocessing.ipynb`)
This notebook handles data cleaning and feature engineering:
- Missing value handling
- Text cleaning (remove URLs, special characters, convert to lowercase)
- Feature engineering (text length, word count, punctuation counts)
- Categorical feature encoding (source domain)
- Numerical feature normalization (tweet numbers using log transformation)
- Train-test split (80/20 with stratification)
- Saving processed datasets for modeling

### 3. Classical Models (`03_classical_models.ipynb`)
Implementation of traditional machine learning approaches:
- Naive Bayes (Multinomial)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Model comparison with comprehensive metrics
- Hyperparameter tuning
- Feature importance analysis

### 4. Deep Learning (`04_deep_learning.ipynb`)
Deep learning models for sequence and pattern recognition:
- LSTM (Long Short-Term Memory) networks
- CNN (1D Convolutional Neural Network)
- Hybrid CNN-LSTM architecture
- Training with callbacks and early stopping
- Model evaluation and training history visualization

### 5. Transformers (`05_transformers.ipynb`)
State-of-the-art transformer-based models:
- DistilBERT (lightweight transformer)
- BERT-base (comprehensive transformer)
- Fine-tuning on the dataset
- Evaluation and comparison with other approaches

## Usage

1. Data Exploration: Run 01_data_exploration.ipynb to understand the dataset characteristics and identify patterns
2. Preprocessing: Run 02_preprocessing.ipynb to clean and prepare the data for modeling
3. Modeling: Run the respective model notebooks (03, 04, 05) to train and evaluate different approaches
4. Comparison: Compare model performances across all approaches to identify the best solution

## Results

All model performance results and visualizations are automatically saved to the `results/` folder when running the notebooks. The results include:

### Model Performance Summary

Best Overall Model: BERT-base (Transformer) - 85.69% accuracy

Top 5 Models by Accuracy:
1. BERT-base (Transformer): 85.69%
2. DistilBERT (Transformer): 85.47%
3. Naive Bayes (Classical ML): 83.73%
4. CNN (Deep Learning): 83.21%
5. Logistic Regression (Classical ML): 80.86%

### Saved Outputs

Each notebook automatically saves:
- Visualizations (results/visualizations/):
  - Confusion matrices for all models
  - Training history plots (for deep learning models)
  - Model comparison charts (accuracy, F1-score, all metrics)
  - Feature importance plots (Random Forest)
  
- Metrics (results/metrics/):
  - Final model comparison CSV with all metrics
  - Detailed metrics for each model
  - Classification reports

All visualizations are saved as high-resolution PNG files (300 DPI) suitable for presentations and reports.

## Custom Modules

src/data_utils.py
Utility functions for data loading, statistical analysis, visualization helpers, and dataset summary reports. These functions support the data exploration notebook.

src/preprocessing.py
Preprocessing functions for text cleaning, feature extraction, categorical encoding, numerical normalization, and data splitting. These functions are used throughout the preprocessing pipeline.

src/models.py
Model training and evaluation utilities including functions for training models, calculating metrics, comparing performance, and generating visualizations. This module supports the modeling notebooks.

## Key Findings

From the exploratory data analysis:
- The dataset is imbalanced with 75.19% real news and 24.81% fake news
- Text length characteristics are very similar between classes, requiring advanced NLP techniques
- Source domain shows strong predictive signal, with some domains having 88-91% fake news rates
- Tweet engagement patterns differ between classes, with fake news showing more viral outliers

## Notes

- The raw dataset is not tracked in git (see `.gitignore`) to save repository space
- Processed datasets can be regenerated by running the preprocessing notebook
- All notebooks are designed to be run sequentially
- Model checkpoints and large model files are excluded from git if they exceed 50MB
- The project uses stratified train-test splits to maintain class distribution

## References

- FakeNewsNet Dataset
- Scikit-learn Documentation
- TensorFlow/Keras Documentation
- Hugging Face Transformers Library

---

## Project Status: Complete

This project has been fully implemented and evaluated. All components are complete:

Completed Components

1. Data Exploration - Comprehensive EDA with visualizations and insights
2. Data Preprocessing - Text cleaning, feature engineering, train-test split
3. Classical ML Models - Naive Bayes, Logistic Regression, SVM, Random Forest
4. Deep Learning Models - LSTM, CNN, CNN-LSTM Hybrid
5. Transformer Models - DistilBERT, BERT-base fine-tuning
6. Model Comparison - Comprehensive evaluation across all 9 models
7. Results Export - All visualizations and metrics automatically saved

Key Achievements

- 9 models implemented and evaluated across 3 different approaches
- Best accuracy achieved: 85.69% (BERT-base)
- Comprehensive comparison with detailed metrics and visualizations
- Production-ready code with automatic result saving
- Complete documentation with realistic summaries and findings

Final Recommendations

For production deployment, DistilBERT is recommended as it achieves 99.7% of BERT's performance (85.47% vs 85.69%) with 54% of the training time and 61% of the parameters, making it more efficient for real-world applications.

For quick prototyping or resource-constrained environments, Naive Bayes provides excellent performance (83.73% accuracy) with training time under 0.01 seconds.
