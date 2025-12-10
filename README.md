# Fake News Detection - NLP Project

A comprehensive machine learning project for detecting fake news using Natural Language Processing techniques. This project implements multiple modeling approaches including classical ML, deep learning, and transformer-based models.

## 📋 Project Overview

This project aims to classify news articles as either real or fake using various NLP and machine learning techniques. The dataset used is from FakeNewsNet, containing news articles with their titles, source domains, tweet engagement metrics, and labels.

## 📁 Project Structure

```
fake-news-detection/
├── data/
│   ├── raw/                 # Original dataset (not tracked or .gitignore large files)
│   └── processed/           # Cleaned CSVs used by notebooks
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_models.ipynb
│   ├── 04_deep_learning.ipynb
│   └── 05_transformers.ipynb
├── src/
│   ├── data_utils.py
│   ├── preprocessing.py
│   └── models.py
├── requirements.txt
├── environment.yml         # Optional conda environment
├── README.md
└── .gitignore
```

## 📊 Dataset

The dataset contains news articles with the following features:
- **title**: Title of the article
- **news_url**: URL of the article
- **source_domain**: Web domain where article was posted
- **tweet_num**: Number of retweets for this article
- **real**: Label column (1 = real, 0 = fake)

**Dataset Source**: FakeNewsNet (cleaned and combined)

## 🚀 Getting Started

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

**Note**: If `pip` command is not found, use `python3 -m pip install -r requirements.txt` or `pip3 install -r requirements.txt`

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

## 📓 Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Load and examine dataset structure
- Analyze class distribution (real vs fake)
- Text statistics and visualizations
- Source domain analysis
- Tweet engagement metrics
- Missing values analysis

### 2. Preprocessing (`02_preprocessing.ipynb`)
- Handle missing values
- Text cleaning (remove URLs, special characters)
- Feature engineering (text length, word count, etc.)
- Encode categorical features
- Normalize numerical features
- Train-test split (80/20)
- Save processed datasets

### 3. Classical Models (`03_classical_models.ipynb`)
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Model comparison and evaluation

### 4. Deep Learning (`04_deep_learning.ipynb`)
- LSTM (Long Short-Term Memory)
- CNN (1D Convolutional Neural Network)
- Hybrid CNN-LSTM
- Model evaluation and comparison

### 5. Transformers (`05_transformers.ipynb`)
- DistilBERT
- BERT-base
- Fine-tuning and evaluation

## 🔧 Usage

1. **Data Exploration**: Run `01_data_exploration.ipynb` to understand the dataset
2. **Preprocessing**: Run `02_preprocessing.ipynb` to clean and prepare the data
3. **Modeling**: Run the respective model notebooks (03, 04, 05) to train and evaluate models
4. **Comparison**: Compare model performances and select the best model

## 📈 Results

Model performance results will be updated after running the modeling notebooks. Each notebook includes:
- Training and validation metrics
- Confusion matrices
- Classification reports
- Model comparison visualizations

## 🛠️ Custom Modules

### `src/data_utils.py`
Utility functions for:
- Data loading
- Statistical analysis
- Visualization helpers
- Dataset summary reports

### `src/preprocessing.py`
Preprocessing functions for:
- Text cleaning
- Feature extraction
- Categorical encoding
- Numerical normalization
- Data splitting

### `src/models.py`
Model training and evaluation utilities (to be implemented in Day 2-3)

## 📝 Notes

- The raw dataset is not tracked in git (see `.gitignore`)
- Processed datasets are saved in `data/processed/`
- All notebooks are designed to be run sequentially
- Model checkpoints and weights are excluded from git (if large)

## 🤝 Contributing

This is a learning project. Feel free to fork, experiment, and improve!

## 📄 License

This project is for educational purposes.

## 🔗 References

- FakeNewsNet Dataset
- Scikit-learn Documentation
- TensorFlow/Keras Documentation
- Hugging Face Transformers

---

**Status**: Day 1 Complete ✅
- Project structure created
- Data exploration notebook ready
- Preprocessing notebook ready
- Supporting files created

**Next Steps**: Day 2-3 will include model implementation and comparison.

