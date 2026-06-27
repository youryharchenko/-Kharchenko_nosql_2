# scripts/02_generate_embeddings.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Шляхи до файлів та константи
INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
BATCH_SIZE = 64

# Крок 7: Перед збереженням переконатися, що директорія існує; за потреби створити її
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Крок 1: Завантажити датасет із файлу data/arxiv_subset.parquet з використанням pandas
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Помилка: Файл {INPUT_FILE} не знайдено. Спочатку підготуйте дані.")

df = pd.read_parquet(INPUT_FILE)
print(f"Успішно завантажено датасет. Кількість записів: {len(df)}")

# Крок 2: Підготувати тексти для кодування
# Об’єднуємо поля title і abstract в один рядок у форматі: title + " [SEP] " + abstract
print("Підготовка текстів для кодування...")
prepared_texts = [
    f"{row['title']} [SEP] {row['abstract']}" 
    for _, row in df.iterrows()
]

# Крок 3: Ініціалізація моделі allenai/specter2_base з бібліотеки sentence-transformers
print("Завантаження моделі allenai/specter2_base...")
model = SentenceTransformer("allenai/specter2_base")

# Крок 4: Закодувати всі тексти в ембеддинги з урахуванням вимог:
# - батчева обробка (batch_size=64)
# - відображення прогресу (show_progress_bar=True)
# - нормалізація ембеддингів (normalize_embeddings=True)
print(f"Початок генерації ембеддингів (batch_size={BATCH_SIZE})...")
embeddings = model.encode(
    prepared_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Крок 5: Вивести в консоль метрики для перевірки
print("\n" + "="*50)
print("АНАЛІЗ ОТРИМАНИХ ЕМБЕДДИНГІВ:")
print("="*50)
# 5.1 Загальна кількість оброблених текстів
print(f"Загальна кількість оброблених текстів: {embeddings.shape[0]}")
# 5.2 Розмірність ембеддингів (очікується 768)
print(f"Розмірність ембеддингів:              {embeddings.shape[1]}")

# 5.3 Норма першого ембеддингу (L2 норма повинна бути близька до 1.0, бо увімкнено normalize)
first_embedding_norm = np.linalg.norm(embeddings[0])
print(f"Норма першого ембеддингу:             {first_embedding_norm:.6f}")
print("="*50 + "\n")

# Крок 6: Зберегти отримані ембеддинги у файл embeddings/embeddings.npy у форматі NumPy
np.save(OUTPUT_FILE, embeddings)
print(f"Процес завершено. Ембеддинги успішно збережено у файл: {OUTPUT_FILE}")