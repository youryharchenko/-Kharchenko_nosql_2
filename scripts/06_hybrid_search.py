# scripts/06_hybrid_search.py
import os
import re
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# Налаштування та константи
INDEX_NAME = "arxiv-papers"
DATA_FILE = "data/arxiv_subset.parquet"
RRF_K = 60  # Стандартна константа згладжування для RRF

# Перевірка API-ключа Pinecone
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY не знайдено в змінних оточення!")

# =====================================================================
# Крок 1: Побудова локального індексу BM25
# =====================================================================
print("Крок 1: Завантаження даних та побудова локального BM25-індексу...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Файл {DATA_FILE} не знайдено.")

df = pd.read_parquet(DATA_FILE)

def tokenize(text: str) -> list:
    """ Проста токенізація: приведення до нижнього регістру та виділення слів """
    return re.findall(r'\w+', text.lower())

# Об'єднуємо назву та анотацію для повнотекстового індексу BM25
corpus_texts = [f"{row['title']} {row['abstract']}" for _, row in df.iterrows()]
tokenized_corpus = [tokenize(text) for text in corpus_texts]

# Ініціалізуємо індекс BM25
bm25 = BM25Okapi(tokenized_corpus)


# =====================================================================
# Крок 2: Підключення до Pinecone та завантаження моделі
# =====================================================================
print("Крок 2: Підключення до Pinecone та завантаження моделі Specter2...")
pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer("allenai/specter2_base")


# =====================================================================
# Крок 3 & 4: Реалізація функцій пошуку та RRF
# =====================================================================

def search_bm25(query: str, top_k: int = 20) -> list:
    """ Лексичний пошук за допомогою BM25 """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    # Беремо топ-K результатів за оцінкою BM25
    top_indices = scores.argsort()[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "id": f"paper_{idx}",
            "rank": rank,
            "score": float(scores[idx]),
            "title": df.iloc[idx]['title'],
            "category": df.iloc[idx]['category'],
            "year": df.iloc[idx]['year'],
            "abstract": df.iloc[idx]['abstract']
        })
    return results

def search_vector(query: str, top_k: int = 20) -> list:
    """ Семантичний пошук через Pinecone """
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    results = []
    for rank, match in enumerate(res['matches'], 1):
        meta = match['metadata']
        results.append({
            "id": match['id'],
            "rank": rank,
            "score": match['score'],
            "title": meta['title'],
            "category": meta['category'],
            "year": meta['year'],
            "abstract": meta['abstract']
        })
    return results

def search_hybrid(query: str, top_k: int = 5) -> list:
    """ Гібридний пошук: об'єднання BM25 та векторного через RRF """
    # Отримуємо ширші списки (наприклад, по 50 кандидатів) з обох систем
    candidates_count = 50
    bm25_res = search_bm25(query, top_k=candidates_count)
    vector_res = search_vector(query, top_k=candidates_count)
    
    rrf_scores = {}
    doc_details = {}
    
    # Обробка результатів BM25
    for item in bm25_res:
        doc_id = item['id']
        rank = item['rank']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank))
        doc_details[doc_id] = item
        
    # Обробка векторних результатів
    for item in vector_res:
        doc_id = item['id']
        rank = item['rank']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank))
        # Якщо документа не було в BM25, додаємо деталі
        if doc_id not in doc_details:
            doc_details[doc_id] = item

    # Сортування за RRF-скором
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    hybrid_results = []
    for doc_id, rrf_score in sorted_docs:
        details = doc_details[doc_id]
        hybrid_results.append({
            "id": doc_id,
            "rrf_score": rrf_score,
            "title": details['title'],
            "category": details['category'],
            "year": details['year'],
            "abstract": details['abstract']
        })
    return hybrid_results


# =====================================================================
# Крок 5 & 6: Демонстрація та порівняння трьох різних запитів
# =====================================================================
test_queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

for query in test_queries:
    print("\n" + "="*100)
    print(f"ЗАПИТ: '{query}'")
    print("="*100)
    
    # Виконуємо три типи пошуку
    res_bm25 = search_bm25(query, top_k=5)
    res_vec = search_vector(query, top_k=5)
    res_hyb = search_hybrid(query, top_k=5)
    
    # Вивід BM25
    print("\n[ТОП-5 ВІД BM25 (Лексичний)]")
    for i, doc in enumerate(res_bm25, 1):
        print(f" {i}. [Score: {doc['score']:.2f}] {doc['title']} ({doc['year']}) [{doc['category']}]")
        
    # Вивід Векторного пошуку
    print("\n[ТОП-5 ВІД ВЕКТОРНОГО ПОШУКУ (Семантичний)]")
    for i, doc in enumerate(res_vec, 1):
        print(f" {i}. [Score: {doc['score']:.4f}] {doc['title']} ({doc['year']}) [{doc['category']}]")
        
    # Вивід Гібридного пошуку
    print("\n[ТОП-5 ВІД ГІБРИДНОГО ПОШУКУ (RRF)]")
    for i, doc in enumerate(res_hyb, 1):
        print(f" {i}. [RRF Score: {doc['rrf_score']:.5f}] {doc['title']} ({doc['year']}) [{doc['category']}]")