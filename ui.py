def get_user_selection(documents):
    print("Available documents:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")

    selected = input(
        "Enter the numbers of the documents you want to process (comma-separated), or 'all' for all documents: ")
    if selected.lower() == 'all':
        return documents
    else:
        selected_indices = [int(x.strip()) - 1 for x in selected.split(',')]
        return [documents[i] for i in selected_indices if 0 <= i < len(documents)]


def run_query_loop(basic_engine, advanced_engine, all_blocks):
    query = input("Введите ваш запрос: ")
    results = basic_engine.semantic_search(query, top_k=3)
    display_results(query, results)
    analyze_results('1', query, results, all_blocks, basic_engine, advanced_engine)

    # while True:
    #     engine_choice = input("Выберите движок (1 - базовый, 2 - продвинутый, 'exit' - выход): ")
    #     if engine_choice.lower() == 'exit':
    #         break
    #
    #     query = input("Введите ваш запрос: ")
    #
    #     # Сначала проверяем неверный ввод, потом проверяем наличие движков
    #     if engine_choice not in ('1', '2'):
    #         print("Неверный выбор. Введите 1 или 2.")
    #         continue
    #
    #     if engine_choice == '1' and basic_engine:
    #         results = basic_engine.semantic_search(query, top_k=5)
    #     elif engine_choice == '2' and advanced_engine:
    #         results = advanced_engine.search(query, k=5)
    #     else:  # Сюда попадем, только если engine_choice - '1' или '2', но соответствующий движок не инициализирован
    #         print(f"Движок {engine_choice} не инициализирован. Выберите другой движок.")
    #         continue
    #
    #     display_results(query, results)
    #     analyze_results(engine_choice, query, results, all_blocks, basic_engine, advanced_engine)

def display_results(query, results):
    print(f"\nЗапрос: {query}")
    for i, (block, score) in enumerate(results, 1):
        print(f"\nРезультат #{i} (оценка: {score:.4f}):")
        print(f"Текст: {block[0] if isinstance(block, tuple) else block}")
        print(f"Источник: {block[1] if isinstance(block, tuple) and len(block) > 1 else 'Unknown'}")


def analyze_results(engine_choice, query, results, all_blocks, basic_engine, advanced_engine):
    print("\nАнализ результатов:")
    if engine_choice == '1':
        key_terms = basic_engine.extract_key_terms(query)
    else:
        key_terms = advanced_engine.preprocess_text(query).split()
    print(len(key_terms))
    for term in key_terms:
        if engine_choice == '1':
            top_count = sum(1 for block, _ in results if term in basic_engine.preprocess_text(block[0]))
            all_count = sum(1 for block in all_blocks if term in basic_engine.preprocess_text(block[0]))
        # else:
        #     top_count = sum(1 for block, _ in results if term in advanced_engine.preprocess_text(block))
        #     all_count = sum(1 for block in all_blocks if term in advanced_engine.preprocess_text(block[0]))

        print(f"Термин '{term}' встречается в {top_count} из {len(results)} топ результатов "
              f"и в {all_count} блоках из {len(all_blocks)}.")
