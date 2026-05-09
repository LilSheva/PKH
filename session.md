# PKH — Session Log

## Сессия 1 (2026-05-09)

### Выполнено

1. **Инициализация проекта.** Создан репозиторий, README, claude.md со спецификацией.
2. **Выбор эмбеддинг-моделей.** Отказались от `all-MiniLM-L6-v2` (англоязычная, устаревшая). Выбраны 3 мультиязычные модели с поддержкой per-model коллекций в Chroma:
   - `BAAI/bge-m3` (дефолт) — окно 8192, мультиязычная
   - `Qwen/Qwen3-Embedding-0.6B` — топ MTEB 2025, сильна на коде
   - `intfloat/multilingual-e5-large-instruct` — лёгкий baseline
3. **Реализация ядра (Этап 1 — Ingestion & Smart Parsing):**
   - `config.py` — все настройки в одном месте
   - `core/cleaner.py` — regex-чистка base64/hex/токенов
   - `core/ingestion.py` — content sniffing (`is_chat_file`), `file_hash`, `iter_files`
   - `core/parser.py` — `load_chat` → `ChatPayload`, `iter_blocks` → `DialogBlock`, THOUGHT_MODE OFF/ON/SMART, рекурсивная очистка `thoughtSignature`
   - `core/embeddings.py` — реестр 3 моделей, lazy-загрузка, `embed_query`/`embed_passages`
   - `core/vector_db.py` — Chroma persistent client, per-model коллекция, upsert, dedup helpers
   - `core/manifest.py` — JSON-манифест файлов (is_chat, mtime, size, hash, system_instruction, topics, tags), атомарное сохранение
   - `core/tagger.py` — LLM-классификация чатов (provider-agnostic), стратегическое сэмплирование (head + middle + tail), промпт на русском
   - `sniper.py` — оркестратор: `ingest()`, `search_context()`, `generate_super_prompt()`, `tag_chats()`, `stats()`
4. **Colab-ноутбук** (`pkh_colab.ipynb`) — 9 секций: монтирование Drive, clone, pip install, GPU check, конфиг, ингест, поиск, супер-промпт, сравнение 3 моделей.
5. **Тесты (30/30 зелёные, без GPU/сети):**
   - `tests/test_parser.py` — 11 тестов (включая прогон по 6 реальным шаблонам)
   - `tests/test_manifest.py` — 8 тестов
   - `tests/test_tagger.py` — 11 тестов (fake-LLM, парсинг JSON, стратегическое сэмплирование)
6. **Валидация на реальных данных.** 6 чатов из `chats teamplates/` (от 72 КБ до 76 МБ) — парсер корректно обрабатывает все, включая мультимодальные (base64 не слурпается).
7. **Архитектурные решения:**
   - Убран MD-парсер (вход только JSON, MD = выходной формат)
   - Убран Слой A дедупликации (SOURCE_PRIORITY) — не нужен без MD на входе
   - Манифест хранит `system_instruction` для будущей LLM-классификации (вариант C)
   - Таггер использует стратегическое сэмплирование (не первые 4000 символов, а head+middle+tail) — учитывает дрифт тем по ходу чата
8. **`.gitignore`** — Python-артефакты, Chroma, манифест, IDE, секреты.

### Ключевые находки по формату AI Studio

- Топ-уровень JSON: `runSettings`, `systemInstruction`, `chunkedPrompt`
- `systemInstruction` — пустой `{}` у всех 6 шаблонов (дефолт AI Studio)
- Чанки содержат: `role` (user/model), `text`, `isThought`, `parts[].text`, `parts[].thought`
- Мультимодальные поля: `driveImage`, `driveDocument`, `inlineImage`, `createTime`, `branchChildren` — игнорируются
- `thoughtSignature` — не встретился в шаблонах, но рекурсивная страховка оставлена
- Роли строго `user`/`model`, чередуются (user → 1+ model)

---

## Предстоящие шаги

### Tier 1 — нужно до реального использования
1. **E2E прогон в Colab.** Проверить: загрузку bge-m3, ChromaDB upsert, THOUGHT_MODE=SMART (реальный эмбеддер), дистанции в `search_context`. Всё протестировано изолированно, но e2e — нет.
2. **Wired Gemini для таггера.** Конкретный `gemini_call(prompt) -> str` через `google-genai` SDK. Ячейка в ноутбуке. ~30 строк.
3. **Прогресс-бар при ингесте.** `tqdm` поверх `iter_files()` — на архиве в сотни чатов критично для UX.

### Tier 2 — улучшения после первого прогона
4. **Зеркалирование тегов в Chroma metadata.** После `tag_chats` пройтись `collection.update(...)` и положить `tags`/`topics` в metadata блоков того же `chat_id`. Тогда `search_context(query, where={"tags": ...})` начнёт работать.
5. **CLI-обёртка.** `pkh ingest`, `pkh tag`, `pkh search "..."`, `pkh prompt "..." --meta "..."`. Файл `cli.py` + `pyproject.toml` с `[project.scripts]`.
6. **Слой C (semantic dedup).** Включить, если на реальной базе увидим почти-дубли.

### Tier 3 — архитектурные доработки
7. **`branchChildren`** — проверить, есть ли реально ветвление в чатах, и нужно ли парсить альтернативные ветки.
8. **`system_instruction` в `generate_super_prompt`** — подмешивать SI чата к найденным блокам как контекст.
9. **`pyproject.toml` + package layout** — вместе с CLI.

### Roadmap (заморожено, не сейчас)
- **PKH Radar** — UMAP + HDBSCAN визуализация кластеров.
- **Persona Snapshot / Unfinished Business** — LLM-аналитика «над чем зависал», «что забросил».
- **Serendipity Widget** — рандомизатор инсайтов на пересечении случайных тегов.
