# scripts/03_populate_vector_db.py
import os
import time
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Константи та шляхи
INDEX_NAME = "arxiv-papers"
DATA_FILE = "data/arxiv_subset.parquet"
EMBEDDINGS_FILE = "embeddings/embeddings.npy"
BATCH_SIZE = 200

# Перевірка наявності API-ключа
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("Помилка: PINECONE_API_KEY не знайдено в змінних оточення!")

# Крок 1: Ініціалізація та створення індексу в Pinecone
print("Підключення до сервісу Pinecone...")
pc = Pinecone(api_key=api_key)

# Перевіряємо, чи існує індекс, якщо ні — створюємо його
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    print(f"Індекс '{INDEX_NAME}' не знайдено. Створюємо новий індекс...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,      # Розмірність для allenai/specter2_base
        metric="cosine",     # Наша обрана косинусна метрика
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Стандартний безкоштовний регіон AWS для Free Tier
        )
    )
    # Чекаємо ініціалізації індексу хмарою Pinecone
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
    print(f"Індекс '{INDEX_NAME}' успішно створено та готово до роботи!")
else:
    print(f"Індекс '{INDEX_NAME}' вже існує. Підключаємось...")

# Підключаємось до індексу
index = pc.Index(INDEX_NAME)

# Крок 2: Завантаження локальних даних
print("\nЗавантаження локальних файлів...")
if not os.path.exists(DATA_FILE) or not os.path.exists(EMBEDDINGS_FILE):
    raise FileNotFoundError("Помилка: Відсутні файли даних або ембеддингів. Виконайте попередні кроки.")

df = pd.read_parquet(DATA_FILE)
embeddings = np.load(EMBEDDINGS_FILE)

if len(df) != len(embeddings):
    raise ValueError("Помилка: Кількість рядків у Parquet та векторів у .npy файлі не збігається!")

total_records = len(df)
print(f"Успішно завантажено {total_records} статей та їхніх ембеддингів.")

# Крок 3 та 4: Підготовка даних та завантаження в Pinecone батчами
print(f"\nПочаток завантаження в Pinecone пакетами по {BATCH_SIZE} елементів...")

# tqmd буде відстежувати загальний прогрес по батчах
for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Upsert у Pinecone"):
    batch_upsert_data = []
    
    # Визначаємо межі поточного батчу
    end_idx = min(i + BATCH_SIZE, total_records)
    
    for idx in range(i, end_idx):
        row = df.iloc[idx]
        vector = embeddings[idx].tolist()  # Перетворюємо масив NumPy у звичайний список Python float'ів
        
        # Обрізаємо текстові поля відповідно до технічних обмежень
        abstract_trimmed = row['abstract'][:500]
        authors_trimmed = str(row['authors'])[:200]
        
        # Формуємо об'єкт для Pinecone: (id, вектор, метадані)
        item = (
            f"paper_{idx}",  # унікальний id вигляду "paper_<номер>"
            vector,          # ембеддинг (list of floats)
            {                # словник метаданих
                "arxiv_id": str(row['id']),
                "title": str(row['title']),
                "abstract": abstract_trimmed,
                "authors": authors_trimmed,
                "year": int(row['year']),
                "category": str(row['category'])
            }
        )
        batch_upsert_data.append(item)
    
    # Виконуємо операцію upsert для сформованого батчу
    index.upsert(vectors=batch_upsert_data)

print("\nВсі дані успішно відправлено в хмару!")

# Крок 5: Виведення фінальної кількості векторів в індексі
# Хмарі Pinecone може знадобитися кілька секунд для повної індексації, тому робимо невелику паузу
print("Оновлення статистики індексу...")
time.sleep(2) 

index_stats = index.describe_index_stats()
print("=" * 50)
print(f"ФІНАЛЬНИЙ СТАН ІНДЕКСУ '{INDEX_NAME}':")
print(f"Загальна кількість векторів в індексі: {index_stats['total_vector_count']}")
print("=" * 50)