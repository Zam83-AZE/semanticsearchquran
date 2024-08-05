import logging
from search_engines import SemanticSearchEngine, AdvancedSemanticSearchEngine
from data_processing import process_selected_books, list_available_documents
from utils import display_processed_books
from config import BOOKS_DIRECTORY, EMBEDDINGS_FILE, BOOK_RECORD_FILE, DEVICE
from ui import get_user_selection, run_query_loop
from data_processing import load_processed_books
import pickle
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticSearchApp:
    def __init__(self):
        self.basic_engine = None
        self.advanced_engine = None
        self.all_blocks = []

    def initialize(self):
        logging.info("Запуск программы...")
        logging.info(f"Используемое устройство: {DEVICE}")

        # 1. Загрузка обработанных данных (независимо от новых книг)
        if os.path.exists(EMBEDDINGS_FILE):
            logging.info("Загрузка ранее обработанных данных...")
            with open(EMBEDDINGS_FILE, 'rb') as f:
                self.all_blocks, block_embeddings = pickle.load(f)

            # Создание движков поиска на основе загруженных данных
            self.basic_engine = SemanticSearchEngine(self.all_blocks, precomputed_embeddings=block_embeddings)
            #self.advanced_engine = AdvancedSemanticSearchEngine([block[0] for block in self.all_blocks])

        # 2. Обработка новых книг (если есть)
        processed_books = load_processed_books(BOOK_RECORD_FILE)
        available_documents = list_available_documents(BOOKS_DIRECTORY, processed_books)

        if available_documents:
            selected_books = get_user_selection(available_documents)

            if selected_books:
                self.all_blocks, block_embeddings, sentence_counts, processed_books, total_sentences = process_selected_books(
                    BOOKS_DIRECTORY, selected_books, EMBEDDINGS_FILE, BOOK_RECORD_FILE
                )

                display_processed_books(processed_books)

                if sentence_counts:
                    logging.info(
                        f"\nВсего обработано предложений в новых или измененных книгах: {sum(sentence_counts.values())}")
                else:
                    logging.info("\nНовых или измененных книг для обработки нет.")

                logging.info(f"Всего предложений во всех обработанных книгах: {total_sentences}")

                # Обновление движков поиска с учетом новых данных
                self.basic_engine = SemanticSearchEngine(self.all_blocks, precomputed_embeddings=block_embeddings)
                self.advanced_engine = AdvancedSemanticSearchEngine([block[0] for block in self.all_blocks])

        return True  # Инициализация всегда успешна, даже если нет новых книг

    def run(self):
        self.initialize()  # Инициализация всегда выполняется

        # Запуск цикла поиска (всегда, так как движки создаются в initialize)
        run_query_loop(self.basic_engine, self.advanced_engine, self.all_blocks)

        print("Программа завершена.")

def main():
    app = SemanticSearchApp()
    app.run()

if __name__ == "__main__":
    main()