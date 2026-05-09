# PKH — Personal Knowledge Hub

**Context Sniper для дампов чатов Google AI Studio.**

Утилита для умного поиска и извлечения релевантного контекста из сырых JSON-дампов чатов Google AI Studio с использованием локальной векторной базы [ChromaDB](https://www.trychroma.com/). Код разрабатывается в Codespaces, исполняется в Google Colab на примонтированном Google Drive.

## Зачем

Накопленные диалоги с моделями — ценная база знаний, но она лежит мёртвым грузом в JSON-файлах с шумом, размышлениями модели и base64-мусором. PKH сканирует этот архив, чистит, векторизует и позволяет вытаскивать релевантные фрагменты по смысловому запросу — чтобы подмешивать их в новый промпт.

## Стек

- Python 3.10+
- [`chromadb`](https://github.com/chroma-core/chroma) — локальная векторная БД
- [`sentence-transformers`](https://www.sbert.net/) — эмбеддинги (см. ниже)
- `pydantic`, `regex`, `numpy`
- **GPU обязательно для ингеста.** В Colab включить `Runtime → Change runtime type → T4 GPU` (бесплатный тариф). На CPU ингест больших архивов будет идти часами; поиск (`search_context`) после построения базы работает быстро и без GPU.

### Эмбеддинг-модели

PKH поддерживает три модели на выбор. Для каждой создаётся **отдельная ChromaDB-коллекция**, поэтому можно ингестить один и тот же архив несколькими моделями параллельно и потом сравнивать качество поиска, не перестраивая базу.

| Ключ в `config.py` | Модель | Dim | Контекст | Когда выбирать |
|---|---|---|---|---|
| `bge-m3` | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | 1024 | 8192 | **Дефолт.** Мультиязычная (ru/en/код), длинное окно — длинные ответы модели и «размышления» режутся минимально. |
| `qwen3` | [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 1024 | 32k | Свежая (2025), топ MTEB, особенно сильна на коде и инструкциях. |
| `e5-instruct` | [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 1024 | 514 | Проверенный мультиязычный baseline, инструкционный (`query:` / `passage:`). Самая лёгкая по VRAM. |

Все три влезают в T4 (16 GB VRAM). Переключение — одной строкой:

```python
# config.py
EMBEDDING_MODEL = "bge-m3"   # bge-m3 | qwen3 | e5-instruct
```

Коллекции в Chroma именуются по схеме `pkh_<ключ>` (`pkh_bge-m3`, `pkh_qwen3`, `pkh_e5-instruct`) — `vector_db.py` сам выбирает нужную по текущему значению `EMBEDDING_MODEL`.

## Структура

```
config.py              # Пути, лимиты, THOUGHT_MODE, EMBEDDING_MODEL
core/
  ingestion.py         # Сканирование файлов через content sniffing
  parser.py            # Парсинг chunkedPrompt, сборка пар User+Model
  cleaner.py           # Regex-очистка мусора
  embeddings.py        # Реестр моделей (bge-m3 / qwen3 / e5-instruct), фабрика загрузчика
  vector_db.py         # ChromaDB + sentence-transformers, коллекция per-модель
  manifest.py          # JSON-манифест файлов (is_chat, mtime, hash, system_instruction, tags)
  tagger.py            # LLM-классификация чатов (provider-agnostic, стратегическое сэмплирование)
sniper.py              # ContextSniper: ingest / search_context / generate_super_prompt / tag_chats
pkh_colab.ipynb        # Colab launcher (монтирование, ингест, поиск, сравнение моделей)
tests/
  test_parser.py       # 11 тестов
  test_manifest.py     # 8 тестов
  test_tagger.py       # 11 тестов
chats teamplates/      # 6 примеров реальных дампов AI Studio
```

## Ключевые принципы

1. **Content sniffing вместо расширений.** Файлы дампов могут быть без расширения или с ложным (`file.asset`). `is_chat_file(path)` читает первые 100 байт и ищет `{` + `"chunkedPrompt"`.
2. **Парсинг чанков.** Извлекается `chunkedPrompt.chunks`, где 1 блок = `[User] + [Model]`. Поля `"thoughtSignature"` всегда отбрасываются.
3. **THOUGHT_MODE — три режима обработки размышлений модели:**
   - `OFF` — игнорировать, оставить только финальный ответ.
   - `ON` — вклеить размышления перед ответом.
   - `SMART` — длинные размышления (>1000 симв.) разбиваются на абзацы, и в контекст попадают только те, у которых cosine similarity с запросом пользователя выше порога (~0.4).
4. **Очистка перед векторизацией.** `clean_text()` убирает непрерывные буквенно-цифровые строки длиннее 100 символов (base64, токены, hex).

## Основная сущность

```python
class DialogBlock(BaseModel):
    chat_id: str          # имя файла или ID чата
    user_text: str        # запрос пользователя
    model_text: str       # ответ модели (с учётом THOUGHT_MODE)
    cleaned_content: str  # склеенный и очищенный текст для базы
```

## Использование

```python
from sniper import ContextSniper

# Модель можно задать в config.py или передать явно
sniper = ContextSniper(embedding_model="bge-m3")
sniper.ingest("/content/drive/MyDrive/ai_studio_dumps")

# Поиск релевантных фрагментов
hits = sniper.search_context("как я настраивал ChromaDB на Colab")

# Сборка супер-промпта с подмешанным контекстом
prompt = sniper.generate_super_prompt(
    main_prompt="напиши обёртку над ингестом",
    meta_query="ингест файлов без расширения",
)

# Сравнить выдачу разных моделей на одном запросе
for model in ("bge-m3", "qwen3", "e5-instruct"):
    s = ContextSniper(embedding_model=model)
    print(model, s.search_context("монтирование google drive"))
```

## Инкрементальный ингест и манифест

База живая: чаты дописываются, появляются новые файлы. Ингест устроен так, чтобы повторный запуск был дешёвым.

- **Манифест файлов** (`DB_DIR/manifest.json`) — единый реестр всего, что мы видели в `CHATS_DIR`: путь → `is_chat`, `mtime`, `size`, `hash`, `checked_at`, `chat_id`, `system_instruction`, и зарезервированные под будущее `topics: []` / `tags: []`. Не-чаты тоже попадают в манифест — чтобы не sniff'ить их повторно при следующем запуске.
- **Дельта-сканирование.** Если `(mtime, size)` файла не изменились — он целиком пропускается без чтения содержимого. Изменился → пересняли, обновили запись, переингестили.
- **Идемпотентность блоков.** ID каждого `DialogBlock` — детерминированный `sha1(chat_id, chunk_index, user_text)`. Дописанное сообщение → новый блок, старые на тех же ID не трогаются.
- **Слой B — точный хеш контента.** Перед `upsert` проверяется `content_hash` (sha1 от нормализованного текста). Дубли отсекаются.
- **Слой C — семантическая дедупликация.** Заглушка под флагом `ENABLE_SEMANTIC_DEDUP` (по умолчанию off). При включении: top-1 запрос в коллекцию, при cosine > 0.97 блок скипается.

Манифест общий для всех эмбеддинг-моделей — `is_chat` от модели не зависит. Запускать `sniper.ingest(...)` можно сколько угодно раз — отработает только дельту.

### Поля под будущее

В записи манифеста зарезервированы `topics: []` и `tags: []`. Сейчас не заполняются. Под них планируется LLM-классификация чатов с тегами вида `#программирование_python`, `#настройка_среды` — для последующей навигации и фильтрации поиска.

Поле `system_instruction` заполняется автоматически из `systemInstruction` дампа AI Studio (если пользователь его задал в чате) — это самый концентрированный сигнал о теме чата и пригодится той же LLM-классификации.

## Дорожная карта (Roadmap) — НЕ РЕАЛИЗОВЫВАТЬ СЕЙЧАС
Эти фичи лежат "на карандаше" для будущих версий (после завершения снайпера):
- **PKH Radar:** Визуализация кластеров смыслов базы (2D/3D scatter plot). Сырое видение, ждет интеграции UMAP+HDBSCAN.
- **Unfinished Business / Persona Snapshot:** LLM-аналитика намерений и задач по базе.
- **Serendipity Widget:** Рандомизатор инсайтов на стыке случайных тем.

## Статус

**Текущая фаза:** Этап 1 завершён (Ingestion & Smart Parsing). Код протестирован локально (30/30), ожидает первого e2e прогона в Colab.

Подробный лог сессий и план: [`session.md`](session.md) | Инструкции для следующей сессии: [`next_session_plan.md`](next_session_plan.md)