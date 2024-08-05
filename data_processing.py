import os
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
from functools import partial
from utils import get_file_hash, timeit
import logging
from collections import defaultdict
from search_engines import SemanticSearchEngine
import numpy as np


def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def process_text_file(file_path):
    logging.info(f"Processing file: {file_path}")
    try:
        if file_path.lower().endswith('.pdf'):
            text = read_pdf(file_path)
        else:  # Assume it's a .txt file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        sentences = sent_tokenize(text)
        blocks = []
        for i in range(0, len(sentences), 3):
            block = ' '.join(sentences[i:i + 3])
            source_name = os.path.basename(file_path)
            blocks.append((block, source_name))

        logging.info(f"Finished processing {os.path.basename(file_path)}: {len(sentences)} sentences")
        return blocks, len(sentences)
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return [], 0


def process_file(filename, directory, processed_books):
    if not filename.lower().endswith(('.txt', '.pdf')):
        return None

    file_path = os.path.join(directory, filename)
    file_hash = get_file_hash(file_path)

    if filename in processed_books and processed_books[filename] == file_hash:
        logging.info(f"Skipping already processed file: {filename}")
        return [], 0, False, filename, file_hash

    logging.info(f"Processing new or modified file: {filename}")
    try:
        blocks, sentence_count = process_text_file(file_path)
        return blocks, sentence_count, True, filename, file_hash
    except Exception as e:
        logging.error(f"Error processing file {filename}: {str(e)}")
        return None


def list_available_documents(directory, processed_books):
    """
    Возвращает список доступных текстовых файлов в указанной директории,
    исключая уже обработанные.

    Args:
        directory (str): Путь к директории.
        processed_books (dict): Словарь с информацией об обработанных книгах.

    Returns:
        list: Список имен файлов.
    """
    available_documents = [f for f in os.listdir(directory) if f.lower().endswith(('.txt', '.pdf'))]
    return [doc for doc in available_documents if doc not in processed_books]


@timeit
def process_selected_books(directory, selected_books, embeddings_file, book_record_file):
    processed_books = load_processed_books(book_record_file)
    all_blocks = []
    sentence_counts = defaultdict(int)
    new_or_modified_books = False
    total_sentences = 0

    # Load existing embeddings if available
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            existing_blocks, existing_embeddings = pickle.load(f)
    else:
        existing_blocks, existing_embeddings = [], []

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(partial(process_file, directory=directory, processed_books=processed_books),
                                         selected_books), total=len(selected_books), desc="Processing files"))

    for result in results:
        if result is not None:
            blocks, sentence_count, is_new, filename, file_hash = result
            all_blocks.extend(blocks)
            if is_new:
                sentence_counts[filename] = sentence_count
                new_or_modified_books = True
                processed_books[filename] = file_hash
            total_sentences += sentence_count

    if new_or_modified_books or not existing_blocks:
        logging.info("Creating new embeddings for new or modified books")
        engine = SemanticSearchEngine(all_blocks)
        new_block_embeddings = engine.block_embeddings.cpu().numpy()

        # Combine existing and new data
        combined_blocks = existing_blocks + all_blocks

        # Check if existing_embeddings is empty
        if isinstance(existing_embeddings, np.ndarray):
            is_empty = existing_embeddings.size == 0
        else:
            is_empty = len(existing_embeddings) == 0

        if not is_empty:
            combined_embeddings = np.vstack([existing_embeddings, new_block_embeddings])
        else:
            combined_embeddings = new_block_embeddings

        save_processed_books(processed_books, book_record_file)
        with open(embeddings_file, 'wb') as f:
            pickle.dump((combined_blocks, combined_embeddings), f)
    else:
        logging.info("No new books to process, using existing embeddings")
        combined_blocks, combined_embeddings = existing_blocks, existing_embeddings

    return combined_blocks, combined_embeddings, sentence_counts, processed_books, total_sentences

def load_processed_books(book_record_file):
    if os.path.exists(book_record_file):
        with open(book_record_file, 'rb') as f:
            return pickle.load(f)
    return {}


def save_processed_books(processed_books, book_record_file):
    with open(book_record_file, 'wb') as f:
        pickle.dump(processed_books, f)