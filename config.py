import os
import torch
from nltk.corpus import stopwords
import pymorphy3

# Directories and file paths
BOOKS_DIRECTORY = 'books'
EMBEDDINGS_FILE = 'all_books_embeddings.pkl'
BOOK_RECORD_FILE = 'processed_books.pkl'
MODEL_SAVE_PATH = './saved_model'

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NLP tools
RUSSIAN_STOPWORDS = set(stopwords.words('russian'))
MORPH = pymorphy3.MorphAnalyzer()

# Model configurations
BASIC_MODEL_NAME = 'DeepPavlov/rubert-base-cased-sentence'
ADVANCED_MODEL_NAME = 'sberbank-ai/sbert_large_nlu_ru'

# Search configurations
TOP_K_RESULTS = 5
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3