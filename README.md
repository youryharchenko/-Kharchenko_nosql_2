# Завдання 2. Семантичний пошук за науковими статтями

## Частина 1 — Підготовка даних і вибір інструментів 

### 1.1. Завантаження і підготовка датасету

```
$ python3 scripts/00_loads_data.py 

Downloading to ~/.cache/kagglehub/datasets/Cornell-University/arxiv/291.archive...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.65G/1.65G [02:46<00:00, 10.7MB/s]
Extracting files...
Path to dataset files: ~/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/291
```

```
$ python3 scripts/01_prepare_data.py 

Читаємо датасет: 10000it [00:00, 79604.02it/s]

Завантажено статей:10000

Розподіл за категоріями (топ-10):
category
astro-ph              1838
hep-th                 680
hep-ph                 671
quant-ph               564
gr-qc                  350
cond-mat.mes-hall      307
cond-mat.str-el        292
cond-mat.mtrl-sci      291
cond-mat.stat-mech     271
math.AG                209
Name: count, dtype: int64

Розподіл за роками:
year
2007    10000
Name: count, dtype: int64

Приклад запису:
{'id': '0704.0001', 'title': 'Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies', 'abstract': 'A fully differential calculation in perturbative quantum chromodynamics is\npresented for the production of massive photon pairs at hadron colliders. All\nnext-to-leading order perturbative contributions from quark-antiquark,\ngluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\nall-orders resummation of initial-state gluon radiation valid at\nnext-to-next-to-leading logarithmic accuracy. The region of phase space is\nspecified in which thecalculation is most reliable. Good agreement is\ndemonstrated with data from the Fermilab Tevatron, and predictions are made for\nmore detailed tests with CDF and DO data. Predictions are shown for\ndistributions of diphoton pairs produced at the energy of the Large Hadron\nCollider (LHC). Distributions of the diphoton pairs from the decay of a Higgs\nboson are contrasted with those produced fromQCD processes at the LHC, showing\nthat enhanced sensitivity to the signal can be obtained with judicious\nselection of events.', 'authors': 'BalázsC., BergerE. L., NadolskyP. M., YuanC. -P.', 'year': 2007, 'category': 'hep-ph'}

Збережено вdata/arxiv_subset.parquet
```

### 1.2. Вибір інструментів

1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?

**Модель розгортання**.

Pinecone: Повністю керований SaaS (Хмара). Немає локальної версії. Все працює на інфраструктурі Pinecone.

Qdrant: Гібридна. Можна запустити локально через Docker, розгорнути в Kubernetes (Self-hosted) або взяти їхню керовану хмару (Qdrant Cloud).

Chroma: Вбудована (Embedded) або Клієнт-Сервер. Працює прямо всередині вашого Python-процесу (як SQLite) або як легкий Docker-контейнер.

**Ліцензія**.

Pinecone: Пропрієтарна (Закритий код).

Qdrant: Open-Source (Apache 2.0).

Chroma: Open-Source (Apache 2.0).

**Продуктивність (Масштабування)**.

Pinecone: Екстремальна. Легко масштабується до мільярдів векторів завдяки архітектурі подів (Pods) або Serverless.

Qdrant: Висока (Написаний на Rust). Швидкий, ефективно споживає RAM. Має гнучкі налаштування індексу HNSW та квантизації (скалярної/продуктової).

Chroma: Середня. Оптимізований для малих та середніх проєктів.

**Коли обирати**

Pinecone обирають, коли не хочемо витрачати час на адміністрування, моніторинг, реплікацію та масштабування бази даних; проєкт стрімко росте від тисяч статей до десятків мільйонів, і потрібна гарантована стабільність без ручного налаштування шардів; витрати на SaaS окупаються швидкістю розробки.

Qdrant обирають, коли дані регулюються правилами безпеки (наприклад, GDPR чи банківська таємниця), і їх заборонено відправляти у сторонню хмару; маєте власні сервери або Kubernetes-кластер, де запустити Docker-контейнер значно дешевше, ніж платити за хмарні тарифи; потрібен потужний механізм фільтрації за метаданими.

Chroma обирають для прототипування, R&D та тестів; коли додаток має працювати локально на машині користувача.


2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

**all-MiniLM-L6-v2** навчалася на загальних інтернет-текстах (Reddit, Вікіпедія, розмовні форуми, новини). Складний науковий термін може викликати втрату контексту.

**specter2_base** побудована на базі архітектури SciBERT і навчена на корпусі наукових публікацій із бази Semantic Scholar. Вона «розуміє» науковий жартон, структуру анотацій та специфічний академічний стиль мовлення.

У документації наведено специфікації завдань (Downstream Tasks), для яких призначена базова модель та її офіційними адаптерами:

**Proximity (allenai/specter2):**

"Encode papers as queries and candidates eg. Link Prediction, Nearest Neighbor Search."

(Кодування статей як запитів та кандидатів для прогнозування зв'язків між публікаціями або пошуку найближчих сусідів).

**Adhoc Query (allenai/specter2_adhoc_query):**

"Encode short raw text queries for search tasks."

(Кодування коротких сирих текстових запитів користувача для пошукових задач).

**Classification (allenai/specter2_classification):**

"Encode papers to feed into linear classifiers as features."

(Генерація векторних ознак із документів для їх подальшої передачі в лінійні класифікатори).

**Regression (allenai/specter2_regression):**

"Encode papers to feed into linear regressors as features."

(Генерація ознак для лінійної регресії, наприклад, прогнозування метрик або числових характеристик статей).

3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

У картці моделі allenai/specter2_base та її офіційній документації на GitHub і HuggingFace як рекомендована метрика для оцінки подібності векторів використовується косинусна схожість (Cosine Similarity).

Оскільки під час тренування за допомогою Contrastive Learning модель навчалася нормалізувати вектори або оптимізувати кути між ними на основі графа цитувань, саме косинусна відстань відображає коректну геометрію її векторного простору.

Коли ви створюється новий індекс у векторній базі даних, параметр metric є фундаментальним і не може бути змінений після створення індексу. Якщо буде виберана неправильна метрика, система семантичного пошуку буде працювати некоректно, повертаючи нерелевантні результати.

### 1.3. Отримання ембеддингів

```
$ python3 scripts/02_embed.py

Успішно завантажено датасет. Кількість записів: 10000
Підготовка текстів для кодування...
Завантаження моделі allenai/specter2_base...
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 754/754 [00:00<00:00, 2.81MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 440M/440M [00:45<00:00, 9.77MB/s]
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 31077.02it/s]
tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 453/453 [00:00<00:00, 1.75MB/s]
vocab.txt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 228k/228k [00:00<00:00, 4.00MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 717k/717k [00:00<00:00, 6.79MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [00:00<00:00, 371B/s]
model.safetensors:   0%|                                                                                                                                                    | 0.00/440M [00:04<?, ?B/s]Початок генерації ембеддингів (batch_size=64)...
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 440M/440M [00:43<00:00, 10.1MB/s]
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [42:55<00:00, 16.41s/it]
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [42:55<00:00,  3.00s/it]
==================================================
АНАЛІЗ ОТРИМАНИХ ЕМБЕДДИНГІВ:
==================================================
Загальна кількість оброблених текстів: 10000
Розмірність ембеддингів:              768
Норма першого ембеддингу:             1.000000
==================================================

Процес завершено. Ембеддинги успішно збережено у файл: embeddings/embeddings.npy
```

1. Чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?

Косинусна схожість (Cosine Similarity) між двома векторами $A$ та $B$ обчислюється за такою формулою:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

Де:

$A \cdot B$ — це скалярний добуток (Dot Product) двох векторів.

$\|A\|$ — це евклідова норма (довжина) вектора $A$.

$\|B\|$ — це евклідова норма (довжина) вектора $B$.

Нормалізація вектора означає, що кожен вектор ділиться на свою довжину, в результаті чого його нова довжина стає рівною строго $1$. Тобто для нормалізованих векторів одиничної довжини виконується рівність:

$$\|A\| = 1$$

$$\|B\| = 1$$

Підставимо ці значення знаменника в базову формулу косинусної схожості:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{1 \times 1} = A \cdot B$$

Знаменник перетворюється на одиницю. Ми отримуємо чисту тотожність:

$$\text{Cosine Similarity}(A, B) = \text{Dot Product}(A, B)$$

## Частина 2 — Завантаження даних і метадані

```
$ python3 scripts/03_load_to_pinecone.py

Підключення до сервісу Pinecone...
Індекс 'arxiv-papers' вже існує. Підключаємось...

Завантаження локальних файлів...
Успішно завантажено 10000 статей та їхніх ембеддингів.

Початок завантаження в Pinecone пакетами по 200 елементів...
Upsert у Pinecone: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:46<00:00,  1.08it/s]

Всі дані успішно відправлено в хмару!
Оновлення статистики індексу...
==================================================
ФІНАЛЬНИЙ СТАН ІНДЕКСУ 'arxiv-papers':
Загальна кількість векторів в індексі: 10000
==================================================
```

## Частина 3 — Пошукові запити

```
$ python3 scripts/04_search.py

Крок 1-2: Ініціалізація інфраструктури...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 29919.58it/s]

Крок 3: Чистий семантичний пошук для запиту: 'teaching machines to recognize objects in pictures'
--------------------------------------------------------------------------------
#1 [Score: 0.8288] Capturing knots in polymers
   Категорія: cond-mat.soft | Рік: 2007
   Анотація:  This paper visualizes a knot reduction algorithm...

#2 [Score: 0.8263] Symbolic sensors : one solution to the numerical-symbolic interface
   Категорія: physics.ins-det | Рік: 2007
   Анотація:  This paper introduces the concept of symbolic sensor as an extension of the
smart sensor one. Then, the links between the physical world and the symbo...

#3 [Score: 0.8256] The Mathematics
   Категорія: math.HO | Рік: 2007
   Анотація:  This is an essay that considering the knowledge structure and language of a
different nature, attempts to build on an explanation of the object of stu...

#4 [Score: 0.8170] Modeling the field of laser welding melt pool by RBFNN
   Категорія: physics.comp-ph | Рік: 2007
   Анотація:  Efficient control of a laser welding process requires the reliable prediction
of process behavior. A statistical method of field modeling, based on
no...

#5 [Score: 0.8146] Why should anyone care about computing with anyons?
   Категорія: quant-ph | Рік: 2007
   Анотація:  In this article we present a pedagogical introduction of the main ideas and
recent advances in the area of topological quantum computation. We give an...


Крок 4: Хмарний пошук із фільтрацією
================================================================================
Приклад A: Запит 'reinforcement learning policy optimization' | Фільтр: рік >= 2021, категорія == cs.LG

Приклад B: Запит 'teaching machines to recognize objects in pictures' | Фільтр: рік <= 2015
  #1 [2007] [cond-mat.soft] Capturing knots in polymers
  #2 [2007] [physics.ins-det] Symbolic sensors : one solution to the numerical-symbolic interface
  #3 [2007] [math.HO] The Mathematics
  #4 [2007] [physics.comp-ph] Modeling the field of laser welding melt pool by RBFNN
  #5 [2007] [quant-ph] Why should anyone care about computing with anyons?

Крок 5: Локальне порівняння математичних метрик схожості
================================================================================
Топ-5 індексів документів (id у датасеті) для запиту:
Ранг   | Cosine Similarity    | Dot Product          | L2 Distance         
---------------------------------------------------------------------------
1      | idx_378 (0.8294)     | idx_378 (0.8294)     | idx_378 (0.5842)    
2      | idx_3350 (0.8260)    | idx_3350 (0.8260)    | idx_3350 (0.5899)   
3      | idx_4115 (0.8254)    | idx_4115 (0.8254)    | idx_4115 (0.5910)   
4      | idx_610 (0.8181)     | idx_610 (0.8181)     | idx_610 (0.6032)    
5      | idx_3181 (0.8142)    | idx_3181 (0.8142)    | idx_3181 (0.6095)   

Назви топових статей за версією Cosine:
 1. Capturing knots in polymers (2007)
 2. Symbolic sensors : one solution to the numerical-symbolic interface (2007)
 3. The Mathematics (2007)
 4. Modeling the field of laser welding melt pool by RBFNN (2007)
 5. Python for Education: Computational Methods for Nonlinear Systems (2007)

Назви топових статей за версією Dot Product:
 1. Capturing knots in polymers (2007)
 2. Symbolic sensors : one solution to the numerical-symbolic interface (2007)
 3. The Mathematics (2007)
 4. Modeling the field of laser welding melt pool by RBFNN (2007)
 5. Python for Education: Computational Methods for Nonlinear Systems (2007)

Назви топових статей за версією L2 Distance:
 1. Capturing knots in polymers (2007)
 2. Symbolic sensors : one solution to the numerical-symbolic interface (2007)
 3. The Mathematics (2007)
 4. Modeling the field of laser welding melt pool by RBFNN (2007)
 5. Python for Education: Computational Methods for Nonlinear Systems (2007)
```

### Пошук з фільтрацією

Приклад A (рік >= 2021 і категорія == cs.LG): Повернув порожній результат.

Приклад B (рік <= 2015): Відпрацював успішно і повернув ті самі статті 2007 року, оскільки вони повністю підпадають під фільтр.

У нашому Parquet-файлі, який обмежений початком датасету arXiv, фізично немає жодної статті, яка була б опублікована після 2021 року та мала категорію cs.LG.

Щоб запаит дійсно почав знаходити статті про Machine Learning та Computer Science, потрібно змінити стратегію вибірки даних у 01_prepare_data.py.

### Метрики схожості на локальних ембеддингах

1. **Cosine Similarity та Dot Product**: Списки індексів (idx_378, idx_3350...) та скори (наприклад, 0.8294) збігаються до останнього знаку. Це доказ того, що батчева нормалізація на етапі генерації пройшла успішно. Довжина векторів дорівнює $1$, тому косинус перетворився на звичайний скалярний добуток.

2. **L2 Distance**: Видає абсолютно ідентичний порядок статей. Перша стаття має найменшу L2-відстань (0.5842), що відповідає найвищому косинусному скору (0.8294). Математичний зв'язок $d_{L2} = \sqrt{2 - 2 \cdot \text{cos}(\theta)}$ підтверджено практикою.

3. Що сталося б, якби ембеддинги не були нормалізовані?

Довжина вектора в текстових моделях часто корелює з технічними нюансами: кількістю слів в анотації, частотою використання рідкісних токенів або специфікою розподілу уваги (attention) моделі.

Якщо ми шукаємо статті через Dot Product без нормалізації, база даних буде постійно піднімати в топ ті статті, які мають просто довші вектори (довші тексти або більше специфічних термінів), навіть якщо за змістом вони менш релевантні. Довжина вектора почне "перебивати" його напрямок.

Для allenai/specter2_base довжина вектора є "сміттєвою" метрикою, а вся корисна інформація закладена виключно в напрямку вектора.

## Частина 4 — Chunking



