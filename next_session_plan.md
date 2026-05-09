# Следующая сессия — план действий

## Контекст

Этап 1 (Ingestion & Smart Parsing) **реализован и протестирован локально** (30/30 тестов). Код готов к первому реальному прогону в Colab. Таггер (`core/tagger.py`) написан, но LLM-провайдер не подключён.

## С чего начать

### 1. Подключить Gemini API к таггеру (~15 мин)

Создать файл `core/providers/gemini.py` (или ячейку в ноутбуке):

```python
import google.generativeai as genai

genai.configure(api_key="...")  # или из env

def gemini_call(prompt: str) -> str:
    resp = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return resp.text
```

Затем в ноутбуке:
```python
sniper.tag_chats(gemini_call, only_untagged=True)
```

### 2. Добавить tqdm в ингест (~5 мин)

В `sniper.py` → `ingest()`:
```python
from tqdm import tqdm
for path in tqdm(list(ingestion.iter_files(root)), desc="scanning"):
    ...
```

Добавить `tqdm` в `requirements.txt`.

### 3. E2E прогон в Colab

1. Открыть `pkh_colab.ipynb`, включить T4 GPU.
2. Прогнать ингест на реальном архиве (начать с `chats teamplates/` — 6 файлов, быстро).
3. Проверить:
   - Модель bge-m3 загрузилась на GPU.
   - `sniper.stats()` показывает ожидаемое количество блоков.
   - `search_context("event_src.ip")` возвращает релевантные результаты с адекватными дистанциями.
   - `generate_super_prompt(...)` собирает читаемый промпт.
   - THOUGHT_MODE=SMART реально фильтрует абзацы (переключить на ON/OFF и сравнить).
4. Прогнать `tag_chats(gemini_call)` — убедиться, что теги адекватные.
5. Если всё ок — прогнать на полном архиве.

### 4. После успешного прогона — зеркалирование тегов

Добавить метод `ContextSniper.sync_tags_to_chroma()`:
- Пройтись по `manifest.chat_paths()`.
- Для каждого `chat_id` с непустыми `tags`/`topics` — обновить metadata всех блоков этого чата в Chroma через `collection.update(...)`.
- После этого `search_context(query, where={"tags": {"$contains": "#siem"}})` заработает.

### 5. CLI-обёртка (опционально, если есть время)

`cli.py` с подкомандами: `ingest`, `tag`, `search`, `prompt`, `stats`. Использовать `argparse` (без лишних зависимостей). Зарегистрировать через `pyproject.toml` → `[project.scripts] pkh = "cli:main"`.

## Что НЕ делать в следующей сессии

- Не трогать Roadmap-фичи (Radar, Persona, Serendipity).
- Не включать Слой C (semantic dedup) — сначала данные, потом решение.
- Не рефакторить в package layout до появления CLI.
- Не парсить `branchChildren` — сначала проверить, есть ли они вообще в реальном архиве.

## Файлы для ознакомления при старте сессии

- `claude.md` — спецификация проекта (правила парсинга, манифест, дедупликация)
- `session.md` — лог выполненных шагов и полный план
- `README.md` — обзор проекта, стек, структура
- `config.py` — все настройки
- `sniper.py` — точка входа, оркестратор
