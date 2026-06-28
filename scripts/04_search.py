# scripts/04_search_and_evaluate.py
import os
import numpy as np
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Налаштування та константи
INDEX_NAME = "arxiv-papers"
DATA_FILE = "data/arxiv_subset.parquet"
EMBEDDINGS_FILE = "embeddings/embeddings.npy"
QUERY_TEXT = "teaching machines to recognize objects in pictures"

# Перевірка ключа Pinecone
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY не знайдено в змінних оточення!")

# =====================================================================
# Крок 1 та 2: Підключення до індексу та завантаження моделі
# =====================================================================
print("Крок 1-2: Ініціалізація інфраструктури...")
pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("allenai/specter2_base")

def encode_query(query: str) -> list:
    """ Кодує запит і повертає нормалізований список float-значень """
    # Для adhoc-запитів архітектура SPECTER2 також передбачає нормалізацію
    emb = model.encode(query, normalize_embeddings=True)
    return emb.tolist()

# Генеруємо вектор запиту
query_vector = encode_query(QUERY_TEXT)


# =====================================================================
# Крок 3: Чистий семантичний пошук у Pinecone
# =====================================================================
print(f"\nКрок 3: Чистий семантичний пошук для запиту: '{QUERY_TEXT}'")
print("-" * 80)

results_pure = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True
)

for rank, match in enumerate(results_pure['matches'], 1):
    meta = match['metadata']
    print(f"#{rank} [Score: {match['score']:.4f}] {meta['title']}")
    print(f"   Категорія: {meta['category']} | Рік: {meta['year']}")
    print(f"   Анотація:  {meta['abstract'][:150]}...\n")


# =====================================================================
# Крок 4: Пошук з фільтрацією метаданих
# =====================================================================
print("\nКрок 4: Хмарний пошук із фільтрацією")
print("=" * 80)

# Приклад A: Reinforcement Learning (запит змінено під тему), cs.LG, останні 5 років (>= 2021)
rl_query_text = "reinforcement learning policy optimization"
rl_query_vector = encode_query(rl_query_text)

print(f"Приклад A: Запит '{rl_query_text}' | Фільтр: рік >= 2021, категорія == cs.LG")
results_filter_a = index.query(
    vector=rl_query_vector,
    top_k=5,
    include_metadata=True,
    filter={
        "year": {"$gte": 2021},
        "category": {"$eq": "cs.LG"}
    }
)
for rank, match in enumerate(results_filter_a['matches'], 1):
    meta = match['metadata']
    print(f"  #{rank} [{meta['year']}] [{meta['category']}] {meta['title']}")

# Приклад B: Старі статті (до 2015 року), будь-яка категорія
print(f"\nПриклад B: Запит '{QUERY_TEXT}' | Фільтр: рік <= 2015")
results_filter_b = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True,
    filter={
        "year": {"$lte": 2015}
    }
)
for rank, match in enumerate(results_filter_b['matches'], 1):
    meta = match['metadata']
    print(f"  #{rank} [{meta['year']}] [{meta['category']}] {meta['title']}")


# =====================================================================
# Крок 5: Локальне порівняння метрик схожості
# =====================================================================
print("\nКрок 5: Локальне порівняння математичних метрик схожості")
print("=" * 80)

df = pd.read_parquet(DATA_FILE)
X = np.load(EMBEDDINGS_FILE)  # Матриця [10000, 768]
q = np.array(query_vector)     # Вектор [768]

# 1. Cosine Similarity
# Оскільки X та q вже нормалізовані, косинус — це просто скалярний добуток
cos_scores = np.dot(X, q)
top5_cos = np.argsort(cos_scores)[::-1][:5]

# 2. Dot Product (на нормалізованих векторах математично тотожний косинусу)
dot_scores = np.dot(X, q)
top5_dot = np.argsort(dot_scores)[::-1][:5]

# 3. L2 (Euclidean) Distance (менша відстань = більша схожість)
l2_distances = np.linalg.norm(X - q, axis=1)
top5_l2 = np.argsort(l2_distances)[:5]  # Сортування за зростанням

# Виведення результатів у вигляді порівняльної таблиці
print(f"Топ-5 індексів документів (id у датасеті) для запиту:")
print(f"{'Ранг':<6} | {'Cosine Similarity':<20} | {'Dot Product':<20} | {'L2 Distance':<20}")
print("-" * 75)
for i in range(5):
    idx_cos = top5_cos[i]
    idx_dot = top5_dot[i]
    idx_l2  = top5_l2[i]
    
    print(f"{i+1:<6} | {f'idx_{idx_cos} ({cos_scores[idx_cos]:.4f})':<20} | {f'idx_{idx_dot} ({dot_scores[idx_dot]:.4f})':<20} | {f'idx_{idx_l2} ({l2_distances[idx_l2]:.4f})':<20}")

print("\nНазви топових статей за версією Cosine:")
for i, idx in enumerate(top5_cos, 1):
    print(f" {i}. {df.iloc[idx]['title']} ({df.iloc[idx]['year']})")

print("\nНазви топових статей за версією Dot Product:")
for i, idx in enumerate(top5_dot, 1):
    print(f" {i}. {df.iloc[idx]['title']} ({df.iloc[idx]['year']})")

print("\nНазви топових статей за версією L2 Distance:")
for i, idx in enumerate(top5_l2, 1):
    print(f" {i}. {df.iloc[idx]['title']} ({df.iloc[idx]['year']})")