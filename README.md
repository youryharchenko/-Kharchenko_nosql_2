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

```
$ python3  scripts/05_chunking.py

Ініціалізація Pinecone та моделі Specter2...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 32819.54it/s]
Обрано 30 статей. Максимальна довжина анотації: 1872 символів.

Обробка та завантаження для стратегії: Fixed-size
Генерація ембеддингів для 299 чанків...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:24<00:00,  4.80s/it]
Завантаження в індекс Fixed-size батчами по 100...
Upsert Fixed-size: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:09<00:00,  3.07s/it]

Обробка та завантаження для стратегії: Semantic
Генерація ембеддингів для 265 чанків...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:20<00:00,  4.14s/it]
Завантаження в індекс Semantic батчами по 100...
Upsert Semantic: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:07<00:00,  2.47s/it]

==========================================================================================
РЕЗУЛЬТАТИ ПОШУКУ ДЛЯ ЗАПИТУ: 'numerical algorithms and simulation methods'
==========================================================================================

--- Стратегія: FIXED-SIZE CHUNKS ---
 #1 [Score: 0.8322] Стаття: Dependence of CMI Growth Rates on Electron Velocity Distributions and
  Perturbation by Solitary Waves
    [Чанк №2] Текст чанка: modeled to closely fit the electron distribution functions observed in the auroral cavity. We systematically varied the model parameters as well as the propagat...
--------------------------------------------------
 #2 [Score: 0.8231] Стаття: NBODY meets stellar population - The HYDE-PARC Project
    [Чанк №4] Текст чанка: the stellar evolution may also influence the dynamics of the system. Hence, if one is interested in simulating what the morphology will look like through a tele...
--------------------------------------------------
 #3 [Score: 0.8206] Стаття: Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
    [Чанк №1] Текст чанка: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phenomena in s...
--------------------------------------------------
 #4 [Score: 0.8194] Стаття: Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
    [Чанк №2] Текст чанка: recurrence lattices. Main tool of our approach is the method of recursive (hierarchical) lattices. We apply the method of dynamical mapping (or recursive lattic...
--------------------------------------------------
 #5 [Score: 0.8137] Стаття: Ages for illustrative field stars using gyrochronology: viability,
  limitations and errors
    [Чанк №5] Текст чанка: stars it cannot currently be used on. We then calibrate the age dependence using the Sun. The errors are propagated to understand their dependence on color and ...
--------------------------------------------------

--- Стратегія: SEMANTIC CHUNKS ---
 #1 [Score: 0.8229] Стаття: Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
    [Чанк №1] Текст чанка: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phenomena in s...
--------------------------------------------------
 #2 [Score: 0.8220] Стаття: Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
    [Чанк №8] Текст чанка: The cooperative phenomena near the critical point are studied and the results obtained based on the conjectures are compared with those of the approximation met...
--------------------------------------------------
 #3 [Score: 0.8209] Стаття: NBODY meets stellar population - The HYDE-PARC Project
    [Чанк №3] Текст чанка: Due to dynamical effects, such as energy equipartition, different masses of stars may populate different regions in the object. Since stars are evolving in mass...
--------------------------------------------------
 #4 [Score: 0.8177] Стаття: Ages for illustrative field stars using gyrochronology: viability,
  limitations and errors
    [Чанк №1] Текст чанка: We here develop an improved way of using a rotating star as a clock, set it using the Sun, and demonstrate that it keeps time well....
--------------------------------------------------
 #5 [Score: 0.8133] Стаття: Ages for illustrative field stars using gyrochronology: viability,
  limitations and errors
    [Чанк №5] Текст чанка: We delineate which stars it cannot currently be used on. We then calibrate the age dependence using the Sun. The errors are propagated to understand their depen...
--------------------------------------------------

==========================================================================================
РЕЗУЛЬТАТИ ПОШУКУ ДЛЯ ЗАПИТУ: 'experimental physics data analysis'
==========================================================================================

--- Стратегія: FIXED-SIZE CHUNKS ---
 #1 [Score: 0.8455] Стаття: Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
    [Чанк №1] Текст чанка: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phenomena in s...
--------------------------------------------------
 #2 [Score: 0.8403] Стаття: Ages for illustrative field stars using gyrochronology: viability,
  limitations and errors
    [Чанк №5] Текст чанка: stars it cannot currently be used on. We then calibrate the age dependence using the Sun. The errors are propagated to understand their dependence on color and ...
--------------------------------------------------
 #3 [Score: 0.8271] Стаття: Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
    [Чанк №9] Текст чанка: behavior and satisfying the scaling laws. The cooperative phenomena near the critical point are studied and the results obtained based on the conjectures are co...
--------------------------------------------------
 #4 [Score: 0.8269] Стаття: The Supernova Channel of Super-AGB Stars
    [Чанк №10] Текст чанка: find for ECSNe a upper limit of ~20% of all supernovae (abridged)....
--------------------------------------------------
 #5 [Score: 0.8224] Стаття: Dependence of CMI Growth Rates on Electron Velocity Distributions and
  Perturbation by Solitary Waves
    [Чанк №2] Текст чанка: modeled to closely fit the electron distribution functions observed in the auroral cavity. We systematically varied the model parameters as well as the propagat...
--------------------------------------------------

--- Стратегія: SEMANTIC CHUNKS ---
 #1 [Score: 0.8413] Стаття: Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges
    [Чанк №1] Текст чанка: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phenomena in s...
--------------------------------------------------
 #2 [Score: 0.8330] Стаття: Ages for illustrative field stars using gyrochronology: viability,
  limitations and errors
    [Чанк №5] Текст чанка: We delineate which stars it cannot currently be used on. We then calibrate the age dependence using the Sun. The errors are propagated to understand their depen...
--------------------------------------------------
 #3 [Score: 0.8308] Стаття: The Supernova Channel of Super-AGB Stars
    [Чанк №8] Текст чанка: Our synthetic approach allows us to explore the uncertainty of this number imposed by uncertainties in the third dredge-up efficiency and ABG mass loss rate. We...
--------------------------------------------------
 #4 [Score: 0.8239] Стаття: Conjectures on exact solution of three - dimensional (3D) simple orthorhombic Ising lattices
    [Чанк №8] Текст чанка: The cooperative phenomena near the critical point are studied and the results obtained based on the conjectures are compared with those of the approximation met...
--------------------------------------------------
 #5 [Score: 0.8220] Стаття: NBODY meets stellar population - The HYDE-PARC Project
    [Чанк №3] Текст чанка: Due to dynamical effects, such as energy equipartition, different masses of stars may populate different regions in the object. Since stars are evolving in mass...
--------------------------------------------------
```

1. Яка стратегія дає більш осмислені чанки?

З погляду змістовної та логічної цілісності, стратегія Semantic Chunking (семантичне розбиття) дає значно більш осмислені чанки.

Хоча у нашому експерименті Fixed-size стратегія подекуди видавала трохи вищий формальний скор схожості (завдяки жорсткому ущільненню слів та перекриттю), семантичний підхід є безумовним переможцем для реальних прикладних задач (особливо RAG).

Semantic chunking — це професійний інженерний підхід, який враховує синтаксис мови і дає вектори з набагато чистішим смисловим наповненням.

2. Чи є випадки розрізаних речень і як це впливає на ембеддинги?

У стратегії Fixed-size chunking (фіксованого розміру) випадки розрізаних речень є не просто можливими, а неминучими. Оскільки цей алгоритм є «сліпим» до синтаксису мови і враховує лише суху кількість слів, він розрізає текст рівно там, де закінчується ліміт (наприклад, 40 слів), навіть якщо це середина слова чи середина речення.

Розрив синтаксичної структури тексту суттєво впливає на математичне відображення тексту у векторному просторі моделі allenai/specter2_base (та будь-яких інших BERT-подібних моделей). Це зумовлено архітектурою трансформерів.

Коли речення розривається, модель SPECTER2 втрачає здатність до повноцінного семантичного узагальнення фрагмента і починає сильніше опиратися на окремі ключові слова, які випадково опинилися всередині чанка.

Оскільки розірвані чанки містять граматичні уривки, їхні ембеддинги починають стягуватися до центру векторного простору моделі. Тобто вектори стають дуже схожими між собою за базовим скором (0.81 - 0.83), але втрачають тонку семантичну різницю. Базі даних стає важче відділити справді релевантний абзац від просто схожого за набором слів.

3. Як розмір overlap впливає на кількість чанків і покриття тексту?

У фіксованій стратегії частково рятує параметр Overlap (у нашому коді це було 10 слів). Коли алгоритм повертається на 10 слів назад, розрізане речення, яке постраждало в кінці першого чанка, отримує шанс бути прочитаним повністю (або майже повністю) на початку другого чанка.

Коли алгоритм робить крок назад на 10 слів, щоб почати наступний чанк, частина слів дублюється. Це призводить до штучного збільшення загальної кількості фрагментів у базі даних (приблизно на 10-15%).

Це створює іншу інженерну проблему — дублювання інформації в індексі Pinecone та додаткові фінансові/обчислювальні витрати на збереження копій тих самих словосполучень.

## Частина 5 — Гібридний пошук

```
$ python3  scripts/06_hybrid_search.py

Крок 1: Завантаження даних та побудова локального BM25-індексу...
Крок 2: Підключення до Pinecone та завантаження моделі Specter2...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 29708.72it/s]

====================================================================================================
ЗАПИТ: 'BERT fine-tuning'
====================================================================================================

[ТОП-5 ВІД BM25 (Лексичний)]
 1. [Score: 19.15] A New Measure of Fine Tuning (2007) [hep-ph]
 2. [Score: 17.53] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
  Constraints and the Largest LEP Higgs Event Excess (2007) [hep-ph]
 3. [Score: 16.74] Fine-Tuning in Brane-antibrane Inflation (2007) [hep-th]
 4. [Score: 13.43] Natural SUSY Dark Matter: A Window on the GUT Scale (2007) [hep-ph]
 5. [Score: 12.46] Stability and hierarchy problems in string inspired braneworld scenarios (2007) [hep-th]

[ТОП-5 ВІД ВЕКТОРНОГО ПОШУКУ (Семантичний)]
 1. [Score: 0.8645] Misere quotients for impartial games: Supplementary material (2007) [math.CO]
 2. [Score: 0.8533] Introduction to Phase Transitions in Random Optimization Problems (2007) [cond-mat.stat-mech]
 3. [Score: 0.8500] Abstract Convexity and Cone-Vexing Abstractions (2007) [math.FA]
 4. [Score: 0.8481] The Compositions of the Differential Operations and Gateaux Directional
  Derivative (2007) [math.CO]
 5. [Score: 0.8473] Experimental local realism tests without fair sampling assumption (2007) [quant-ph]

[ТОП-5 ВІД ГІБРИДНОГО ПОШУКУ (RRF)]
 1. [RRF Score: 0.02921] A New Measure of Fine Tuning (2007) [hep-ph]
 2. [RRF Score: 0.02651] Fine-Tuning in Brane-antibrane Inflation (2007) [hep-th]
 3. [RRF Score: 0.02255] On the choice of coarse variables for dynamics (2007) [nlin.CD]
 4. [RRF Score: 0.01639] Misere quotients for impartial games: Supplementary material (2007) [math.CO]
 5. [RRF Score: 0.01613] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
  Constraints and the Largest LEP Higgs Event Excess (2007) [hep-ph]

====================================================================================================
ЗАПИТ: 'Yann LeCun convolutional networks'
====================================================================================================

[ТОП-5 ВІД BM25 (Лексичний)]
 1. [Score: 13.51] On Punctured Pragmatic Space-Time Codes in Block Fading Channel (2007) [cs.IT]
 2. [Score: 13.18] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
  Codes (2007) [cs.IT]
 3. [Score: 8.10] Response of degree-correlated scale-free networks to stimuli (2007) [cond-mat.dis-nn]
 4. [Score: 7.97] Optimization in Gradient Networks (2007) [cond-mat.stat-mech]
 5. [Score: 7.77] Simulation of Robustness against Lesions of Cortical Networks (2007) [q-bio.NC]

[ТОП-5 ВІД ВЕКТОРНОГО ПОШУКУ (Семантичний)]
 1. [Score: 0.8479] Multilayer Perceptron with Functional Inputs: an Inverse Regression
  Approach (2007) [math.ST]
 2. [Score: 0.8431] The Netsukuku network topology (2007) [cs.NI]
 3. [Score: 0.8429] The Compositions of the Differential Operations and Gateaux Directional
  Derivative (2007) [math.CO]
 4. [Score: 0.8346] Modeling the field of laser welding melt pool by RBFNN (2007) [physics.comp-ph]
 5. [Score: 0.8314] Adaptive classification of temporal signals in fixed-weights recurrent
  neural networks: an existence proof (2007) [math.OC]

[ТОП-5 ВІД ГІБРИДНОГО ПОШУКУ (RRF)]
 1. [RRF Score: 0.03078] Optimization in Gradient Networks (2007) [cond-mat.stat-mech]
 2. [RRF Score: 0.02569] Simulation of Robustness against Lesions of Cortical Networks (2007) [q-bio.NC]
 3. [RRF Score: 0.02424] DIA-MCIS. An Importance Sampling Network Randomizer for Network Motif
  Discovery and Other Topological Observables in Transcription Networks (2007) [q-bio.QM]
 4. [RRF Score: 0.01639] On Punctured Pragmatic Space-Time Codes in Block Fading Channel (2007) [cs.IT]
 5. [RRF Score: 0.01639] Multilayer Perceptron with Functional Inputs: an Inverse Regression
  Approach (2007) [math.ST]

====================================================================================================
ЗАПИТ: 'making computers understand human emotions from text'
====================================================================================================

[ТОП-5 ВІД BM25 (Лексичний)]
 1. [Score: 21.88] On the Development of Text Input Method - Lessons Learned (2007) [cs.CL]
 2. [Score: 17.09] An Automated Evaluation Metric for Chinese Text Entry (2007) [cs.HC]
 3. [Score: 16.72] Towards Understanding the Origin of Genetic Languages (2007) [q-bio.GN]
 4. [Score: 12.18] Detecting anchoring in financial markets (2007) [q-fin.TR]
 5. [Score: 11.88] Maximal C*-algebras of quotients and injective envelopes of C*-algebras (2007) [math.OA]

[ТОП-5 ВІД ВЕКТОРНОГО ПОШУКУ (Семантичний)]
 1. [Score: 0.8287] Opinion Dynamics and Sociophysics (2007) [physics.soc-ph]
 2. [Score: 0.8228] On the Development of Text Input Method - Lessons Learned (2007) [cs.CL]
 3. [Score: 0.8092] Extracting the hierarchical organization of complex systems (2007) [physics.soc-ph]
 4. [Score: 0.8028] Novelty and Collective Attention (2007) [cs.CY]
 5. [Score: 0.8021] Narratives within immersive technologies (2007) [cs.HC]

[ТОП-5 ВІД ГІБРИДНОГО ПОШУКУ (RRF)]
 1. [RRF Score: 0.03252] On the Development of Text Input Method - Lessons Learned (2007) [cs.CL]
 2. [RRF Score: 0.02932] Detecting anchoring in financial markets (2007) [q-fin.TR]
 3. [RRF Score: 0.02565] The social aspects of quantum entanglement (2007) [physics.pop-ph]
 4. [RRF Score: 0.02556] An Automated Evaluation Metric for Chinese Text Entry (2007) [cs.HC]
 5. [RRF Score: 0.02457] Information diffusion epidemics in social networks (2007) [physics.soc-ph]
```

| Критерій пошуку | BM25 | Векторний пошук | Гібридний (RRF) |
| --- | --- | --- | --- |
| **Пошук за абревіатурами, кодами, іменами** | 🟢 Ідеально | 🟡 Посередньо | 🟢 Сильно |
| **Розуміння синонімів та контексту** | 🔴 Нуль | 🟢 Ідеально | 🟢 Сильно |
| **Стійкість до "сміття" у видачі** | 🟡 Середня | 🔴 Низька | 🟢 Висока |

1. Який метод дав кращий результат і чому?

За критерієм реальної користі для користувача, то найкращий результат показав Гібридний пошук (BM25 + Векторний через RRF). Кожен із трьох методів у цьому експерименті розкрив свої сильні та слабкі сторони, аномально викривлені нашою вибіркою 2007 року.

BM25 (Лексичний пошук). Шукав точні збіги букв і слів.

Векторний пошук (Семантичний через Pinecone). Шукав математичну близькість концептів у векторному просторі specter2_base.

Гібридний пошук через RRF. Математично об'єднав ранги обох методів за формулою reciprocal rank fusion.

Гібридний пошук виграє завдяки ефекту взаємної компенсації помилок.

2. Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів, і чому?

Наприклад запит 1: 'BERT fine-tuning':

Векторний пошук (Top-5): Не містить статтю про динаміку систем.

BM25 (Top-5): Не містить статтю про динаміку систем.

Гібридний пошук (Top-5): На 3-му місці з'являється стаття "On the choice of coarse variables for dynamics (2007)" з RRF-скором 0.02255.

Алгоритм RRF працює за принципом "консенсусу двох експертів":

Якщо один експерт (BM25) каже, що документ ідеальний (ранг 1), а другий (Вектор) каже, що це взагалі не з тієї опери (ранг 500), гібрид поставиться до документа з підозрою.

Але якщо обидва експерти незалежно один від одного ставлять документ на 6-те та 7-ме місця, це означає, що документ містить і точні слова, і глибокий правильний зміст. Для RRF це сигнал вищої надійності, тому такий прихований кандидат піднімається в топ.

3. Як зміна параметра k в RRF впливає на видачу (наприклад, k=60 vs k=1)?

Параметр $k$ у формулі Reciprocal Rank Fusion (RRF) 

$$RRF\_Score(d) = \frac{1}{k + \text{rank}_{BM25}(d)} + \frac{1}{k + \text{rank}_{Vector}(d)}$$

виконує роль регулятора чутливості до високих рангів. Він визначає, наскільки радикально система карає документи за падіння в рейтингу або, навпаки, наскільки сильно вона згладжує різницю між лідерами та аутсайдерами.

Зміна параметра з $k=60$ (класичний інженерний стандарт) на $k=1$ (екстремальне значення) міняє логіку роботи пошуку.

Зменшуючи $k \to 1$, ми робимо систему жорсткішою, довіряючи топ-1 позиціям окремих алгоритмів.

Збільшуючи $k \to 60$ (або навіть $100$), ми робимо систему демократичнішою, піднімаючи в топ документи, щодо яких лексичний та семантичний пошук дійшли згоди.

## Частина 6 — Аналіз і висновки

1. Семантичний пошук vs BM25. Наведіть конкретні приклади запитів із вашої роботи, де кожен метод виграв. Сформулюйте загальне правило: для яких типів запитів варто надати перевагу кожному з них?

Виграв BM25:

У запиті 'BERT fine-tuning' лексичний пошук чітко відпрацював як дзеркало: він знайшов статті, де фізично були написані слова "Fine Tuning".

Якщо користувач шукає саме рядок "fine-tuning", BM25 гарантує його наявність у тексті. Векторний пошук у цьому ж запиті видав абстрактну математику, де цих слів узагалі не було, оскільки SPECTER2 намагався знайти семантичний збіг для "BERT" у базі 2007 року, де його не існує, і збився зі шляху.

Виграв Семантичний:

У запиті 'Yann LeCun convolutional networks'. Оскільки прізвища LeCun у базі не було, BM25 здався і почав видавати випадкові мережі (наприклад, коди для каналів зв'язку cs.IT).

Чому це перемога семантики? Векторний пошук "зрозумів" концепцію запиту і повернув статті про багатошарові перцептрони (math.ST) та радіально-базисні нейромережі (physics.comp-ph). Тобто він зорієнтувався в домені штучного інтелекту, навіть коли точних слів-маркерів не було в датасеті.

Гібрид показав абсолютну синергію:

У запиті 'making computers understand human emotions from text'.
Це довге формулювання розмило точність BM25, а семантичний пошук повів у бік соціофізики. Гібрид через RRF знайшов статтю "On the Development of Text Input Method - Lessons Learned", яка ідеально збалансувала в собі лінгвістичну сторону (від BM25) та людино-машинну взаємодію (від семантики).

2. Вплив розміру чанка. Що відбувається з якістю пошуку, якщо чанк занадто маленький (10–15 слів)? Якщо занадто великий (500+ слів)? Чи є оптимальний розмір або він залежить від задачі?

Розмір чанка (Chunk Size) — це один із найголовніших гіперпараметрів при побудові векторного пошуку. Зміна цього параметра безпосередньо впливає на математичні властивості ембеддингу.

Якщо чанк занадто маленький (10–15 слів) — це зазвичай одне коротке речення або навіть обірвана фраза. Модель не може зчитати загальну тему документа. 

Якщо чанк занадто великий (500+ слів) — це 2–3 сторінки тексту, або ціла анотація разом із декількома розділами статті. Ембеддинг-моделі (зокрема specter2_base або all-MiniLM-L6-v2) мають фіксовану розмірність вихідного вектора (наприклад, 768 чисел). Втиснути в ці 768 чисел зміст трьох сторінок тексту математично неможливо.

Універсального "золотого" числа не існує — розмір чанка повністю залежить від архітектури вашої задачі та типу запитів. Проте існують перевірені практикою інженерні стандарти:

* Для технічних та наукових текстів: Оптимальним вважається розмір 100–250 слів з перекриттям (overlap) у 20–40 слів (або відповідний семантичний розріз по 3–5 речень).

Для юридичних документів та кодексів: Часто використовують менші чанки (50–100 слів), оскільки кожен пункт чи стаття закону — це атомарне, самодостатнє правило.

3. Невідповідна метрика. Що сталося б, якби ми створили індекс Pinecone з метрикою euclidean (L2), але використовували модель, яка повертає нормалізовані вектори? Обґрунтуйте відповідь математично: виведіть зв’язок між L2 і cosine для одиничних векторів.

Якби ми створили індекс у Pinecone з метрикою euclidean (L2 Distance), але завантажували б туди нормалізовані вектори (одиничної довжини, як у нашому випадку з normalize_embeddings=True), то з погляду якості пошуку не змінилося б абсолютно нічого.

$$\text{L2\_Distance} = \|A - B\| = \sqrt{2 \cdot (1 - \text{Cosine}(A, B))}$$

Якщо вектори нормалізовані, геометрично вони всі лежать на поверхні однієї гіперсфери. Шлях по прямій крізь сферу (L2) і кут по дузі сфери (Cosine) повністю синхронні. Але для економії тактів процесора та грошей на інфраструктуру, за умови нормалізації векторів, завжди обирають metric="dotproduct" або metric="cosine".

4. Обмеження Pinecone Starter. З якими обмеженнями безкоштовного тіру ви зіткнулися (або могли б зіткнутися)? Як би ви вирішили задачу, якби датасет був не 10000, а 10 мільйонів статей?

У нашому експерименті з 10 000 статей (і відповідно 260–300 тисячами чанків у попередньому кроці) ми підійшли впритул до лімітів безкоштовної пісочниці.

Ми зіткнулися з тим, що після апсерту (upsert) семантичний індекс повернув пусту відповідь. Безкоштовні Serverless-індекси використовують загальний пул обчислювальних ресурсів, тому побудова внутрішніх графів наближеного пошуку (HNSW) відбувається з низьким пріоритетом. Нам довелося ставити штучну паузу в 30 секунд.

Безкоштовний тариф Serverless зазвичай обмежує розмір збережених векторів (еквівалент кількох сотень мегабайт або до 100 000–500 000 векторів залежно від розмірності). Якби ми завантажили не 30 статей, а всі 10 000 статей, розбивши їх на чанки, ми б гарантовано перевищили ліміт безкоштовного сховища.

Масштабування системи з $10^4$ до $10^7$ документів вимагає повної відмови від аматорських скриптів та переходу на промислову Big Data архітектуру. Для обробки 10 мільйонів статей знадобиться комплексний підхід.

* Перехід на платні індекси Pinecone з підтримкою Scalar Quantization (SQ) або Binary Quantization (BQ).

* Використання розподілених систем обробки даних (наприклад, Apache Spark або Ray) у хмарі (AWS EMR / GCP Dataproc).

* Використання інструментів масового імпорту. У Pinecone для великих масштабів існує функція Bulk Import (завантаження векторів безпосередньо з файлів у сховищі AWS S3 або GCP Cloud Storage).

* Заміна локального BM25 на повноцінний пошуковий рушій — Elasticsearch або OpenSearch.

