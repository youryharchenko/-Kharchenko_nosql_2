# scripts/05_chunked_indexing_and_search.py
import os
import time
import re
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Налаштування
DATA_FILE = "data/arxiv_subset.parquet"
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
BATCH_SIZE = 100

# Перевірка API-ключ Pinecone
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY не знайдено в змінних оточення!")

# Ініціалізація інфраструктури
print("Ініціалізація Pinecone та моделі Specter2...")
pc = Pinecone(api_key=api_key)
model = SentenceTransformer("allenai/specter2_base")

# =====================================================================
# Крок 1: Вибір 30 статей із найдовшими анотаціями
# =====================================================================
df = pd.read_parquet(DATA_FILE)
df['abstract_len'] = df['abstract'].str.len()
top_30_longest = df.nlargest(30, 'abstract_len').copy()
print(f"Обрано 30 статей. Максимальна довжина анотації: {top_30_longest['abstract_len'].max()} символів.")

# =====================================================================
# Крок 2: Стратегії чанкінгу
# =====================================================================

def fixed_size_chunking(text: str, chunk_size: int = 40, overlap: int = 10) -> list:
    """ Розбиття на фіксовану кількість слів із перекриттям """
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        return [" ".join(words)]
        
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if end >= len(words):
            break
    return chunks

def semantic_chunking(text: str, max_words: int = 45) -> list:
    """ Об'єднання речень до досягнення максимальної кількості слів """
    # Наївний спліттер по реченнях (крапка, знак оклику/питання + пробіл)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # Якщо одне речення саме по собі задовге, додаємо його як окремий чанк
        if sentence_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(sentence)
            continue
            
        if current_word_count + sentence_word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = sentence_words
            current_word_count = sentence_word_count
        else:
            current_chunk.extend(sentence_words)
            current_word_count += sentence_word_count
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# =====================================================================
# Крок 3: Створення окремих індексів у Pinecone
# =====================================================================
def ensure_index(index_name):
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        print(f"Створення індексу {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
ensure_index(INDEX_FIXED)
ensure_index(INDEX_SEMANTIC)

idx_fixed_client = pc.Index(INDEX_FIXED)
idx_semantic_client = pc.Index(INDEX_SEMANTIC)

# =====================================================================
# Крок 4-5: Генерація ембеддингів та батчевий Upsert
# =====================================================================
def process_and_upsert(df_papers, chunk_strategy, pinecone_index, strategy_name):
    print(f"\nОбробка та завантаження для стратегії: {strategy_name}")
    
    all_chunks_data = []
    
    # Крок 4: Складання списку всіх чанків та підготовка текстів для моделі
    for _, row in df_papers.iterrows():
        abstract = row['abstract']
        chunks = chunk_strategy(abstract)
        
        for chunk_num, chunk_text in enumerate(chunks, 1):
            # Додаємо заголовок до кожного чанка для збереження контексту моделі SPECTER
            model_input_text = f"{row['title']} [SEP] {chunk_text}"
            
            all_chunks_data.append({
                "model_input": model_input_text,
                "chunk_text": chunk_text,
                "chunk_num": chunk_num,
                "paper_row": row
            })
            
    # Масова генерація ембеддингів за допомогою нашої моделі
    print(f"Генерація ембеддингів для {len(all_chunks_data)} чанків...")
    model_inputs = [c['model_input'] for c in all_chunks_data]
    embeddings = model.encode(model_inputs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    # Крок 5: Завантаження батчами
    print(f"Завантаження в індекс {strategy_name} батчами по {BATCH_SIZE}...")
    for i in tqdm(range(0, len(all_chunks_data), BATCH_SIZE), desc=f"Upsert {strategy_name}"):
        batch_vectors = []
        end_idx = min(i + BATCH_SIZE, len(all_chunks_data))
        
        for idx in range(i, end_idx):
            item = all_chunks_data[idx]
            row = item['paper_row']
            vector = embeddings[idx].tolist()
            
            unique_id = f"paper_{row['id']}_ch_{item['chunk_num']}"
            
            metadata = {
                "arxiv_id": str(row['id']),
                "title": str(row['title']),
                "chunk_text": str(item['chunk_text']),
                "chunk_num": int(item['chunk_num']),
                "year": int(row['year']),
                "category": str(row['category'])
            }
            batch_vectors.append((unique_id, vector, metadata))
            
        pinecone_index.upsert(vectors=batch_vectors)

# Запускаємо процес побудови обох баз даних
process_and_upsert(top_30_longest, fixed_size_chunking, idx_fixed_client, "Fixed-size")
process_and_upsert(top_30_longest, semantic_chunking, idx_semantic_client, "Semantic")

# Даємо індексам оновитися в хмарі
time.sleep(30)

# =====================================================================
# Крок 6: Функція семантичного пошуку по чанках
# =====================================================================
def search_chunks(query_text: str):
    # Кодуємо запит
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()
    
    print("\n" + "="*90)
    print(f"РЕЗУЛЬТАТИ ПОШУКУ ДЛЯ ЗАПИТУ: '{query_text}'")
    print("="*90)
    
    for name, client in [("FIXED-SIZE CHUNKS", idx_fixed_client), ("SEMANTIC CHUNKS", idx_semantic_client)]:
        print(f"\n--- Стратегія: {name} ---")
        res = client.query(vector=query_vector, top_k=5, include_metadata=True)
        
        for rank, match in enumerate(res['matches'], 1):
            meta = match['metadata']
            print(f" #{rank} [Score: {match['score']:.4f}] Стаття: {meta['title']}")
            print(f"    [Чанк №{meta['chunk_num']}] Текст чанка: {meta['chunk_text'][:160]}...")
            print("-" * 50)

# Тестуємо пошук на кількох різних за змістом запитах
search_chunks("numerical algorithms and simulation methods")
search_chunks("experimental physics data analysis")