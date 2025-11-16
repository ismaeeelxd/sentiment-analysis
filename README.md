# Sentiment Analysis Project

A comprehensive sentiment analysis application that uses machine learning to classify text as positive or negative sentiment. The project includes a user-friendly GUI interface and supports multiple machine learning models for accurate sentiment prediction.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [Technical Details](#technical-details)
- [Requirements](#requirements)

## ✨ Features

- **Multiple ML Models**: Choose from 5 different machine learning algorithms
  - Linear SVM
  - SVM (Sigmoid Kernel)
  - Naive Bayes
  - Logistic Regression
  - Random Forest

- **Interactive GUI**: User-friendly Tkinter-based graphical interface for real-time sentiment analysis

- **Advanced Text Preprocessing**: 
  - Custom tokenization with lemmatization
  - Stop word removal (preserving negation words)
  - Part-of-speech tagging
  - TF-IDF vectorization with unigrams and bigrams

- **Model Evaluation**: Comprehensive evaluation with cross-validation and accuracy metrics

- **Pre-trained Models**: Ready-to-use saved models for immediate sentiment analysis

## 📁 Project Structure

```
sentiment-analysis/
├── assets/
│   └── frame0/              # GUI assets (images, buttons)
├── data/
│   └── raw/
│       ├── pos/             # Positive review samples (1000 files)
│       └── neg/             # Negative review samples (1000 files)
├── saved_models/            # Pre-trained ML models
│   ├── model_linear_svm.pkl
│   ├── model_svm_(sigmoid).pkl
│   ├── model_naive_bayes.pkl
│   ├── model_logistic_regression.pkl
│   └── model_random_forest.pkl
├── test/
│   └── raw/                 # Test data samples
│       ├── pos/
│       └── neg/
├── gui.py                   # Main GUI application
├── preprocess.py            # Model training script
├── helperfun.py             # Text preprocessing utilities
├── test.py                  # Model testing script
├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
└── shuffled_documents.pkl   # Shuffled training data
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "Sentiment analysis project/sentiment-analysis"
```

### Step 2: Install Required Packages

```bash
pip install nltk scikit-learn pandas matplotlib seaborn tkinter
```

### Step 3: Download NLTK Data

Run Python and download the required NLTK datasets:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```

Alternatively, uncomment the download lines in `preprocess.py` (lines 17-22) and run it once.

## 💻 Usage

### Running the GUI Application

To use the pre-trained models for sentiment analysis:

```bash
python gui.py
```

**How to use the GUI:**
1. Enter or paste the text you want to analyze in the text box
2. Select one of the available machine learning models using the checkboxes
3. Click the "Predict" button
4. View the sentiment prediction (Positive or Negative) at the bottom

### Training New Models

To train the models from scratch:

```bash
python preprocess.py
```

This script will:
- Load and preprocess the training data
- Train all 5 machine learning models
- Evaluate models with cross-validation
- Display accuracy metrics and visualization
- Save trained models to `saved_models/` directory

**Note**: The script uses pre-shuffled data from `shuffled_documents.pkl`. If you want to shuffle fresh data, uncomment lines 32-36 in `preprocess.py`.

### Testing Models

To test models on custom data:

```bash
python test.py
```

Modify the `text` variable in `test.py` to point to your test data directory.

## 🤖 Models

The project implements and compares 5 machine learning models:

| Model | Description |
|-------|-------------|
| **Linear SVM** | Support Vector Machine with linear kernel - fast and efficient |
| **SVM (Sigmoid)** | Support Vector Machine with sigmoid kernel - non-linear classification |
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem with multinomial distribution |
| **Logistic Regression** | Linear model for binary classification with probability estimates |
| **Random Forest** | Ensemble method using 300 decision trees with bootstrap sampling |

All models are evaluated using:
- Test set accuracy (20% holdout)
- Cross-validation accuracy (10-fold)
- Training error metrics

## 📊 Data

### Training Data

- **Location**: `data/raw/`
- **Structure**: 
  - `pos/`: 1000 positive review text files
  - `neg/`: 1000 negative review text files
- **Format**: Each file contains a single review in plain text (.txt format)

### Data Preprocessing

The preprocessing pipeline includes:

1. **Tokenization**: Word tokenization using NLTK
2. **Lowercasing**: Convert all text to lowercase
3. **Stop Word Removal**: Remove common stop words (except negation words like "not", "no", "never")
4. **Lemmatization**: Reduce words to their root forms using POS tagging
5. **TF-IDF Vectorization**: 
   - Unigrams and bigrams (1-2 word combinations)
   - Minimum document frequency: 3
   - Maximum document frequency: 95%
   - Sublinear TF scaling

## 🔧 Technical Details

### Text Preprocessing

The `helperfun.py` module provides:

- **Custom Tokenizer**: Implements lemmatization with POS tagging for accurate word reduction
- **Stop Word Filtering**: Preserves negation words crucial for sentiment analysis
- **Data Loading**: Efficient loading of labeled text files from directory structure

### Model Training

The `preprocess.py` script:

- Splits data into 80% training and 20% testing sets
- Uses stratified splitting to maintain class balance
- Applies TF-IDF vectorization with custom tokenizer
- Trains all models and evaluates performance
- Generates visualization comparing model accuracies
- Saves models and vectorizer for future use

### GUI Application

The `gui.py` application:

- Loads pre-trained models and vectorizer on startup
- Provides interactive text input area
- Allows model selection via checkboxes
- Displays real-time sentiment predictions
- Features a modern, user-friendly interface

## 📦 Requirements

### Python Packages

```
nltk>=3.8
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### NLTK Data

- punkt (tokenizer)
- stopwords (stop word lists)
- wordnet (lemmatization)
- omw-1.4 (Open Multilingual Wordnet)
- averaged_perceptron_tagger (POS tagging)

## 📝 Notes

- The project uses pre-shuffled data stored in `shuffled_documents.pkl` for reproducibility
- Models are saved as pickle files for quick loading
- The TF-IDF vectorizer must be saved and loaded with the same models to ensure compatibility
- All models are trained on the same preprocessed data for fair comparison

## 🎯 Future Enhancements

Potential improvements for the project:

- Support for neutral sentiment classification
- Real-time model performance comparison
- Batch text processing capabilities
- Export predictions to file
- Model confidence scores
- Support for additional languages
- Deep learning model integration

## 📄 License

This project is open source and available for educational and research purposes.

## 👤 Author

Created as part of a sentiment analysis project demonstrating machine learning techniques for text classification.

---

**Happy Analyzing! 🎉**

