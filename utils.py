import time
import logging
import hashlib


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result

    return wrapper


def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def display_processed_books(processed_books):
    logging.info("\nProcessed Books:")
    for filename, file_hash in processed_books.items():
        logging.info(f"{filename}: {file_hash}")
