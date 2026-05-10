# Следующая сессия — план действий

## Контекст

Этап 1 (Ingestion & Smart Parsing) **реализован и протестирован** (30/30 тестов). E2E прогон в Colab **выполнен** — ингест 7 чатов (189 блоков), поиск и super-prompt работают. Выявлены проблемы с качеством super-prompt (thoughts в контексте, отсутствие метаданных, нет порога отсечения).

## С чего начать — доработка super-prompt

### 1. Порог отсечения по дистанции

В `generate_super_prompt()` и `search_context()` — не включать блоки с `distance > MAX_CONTEXT_DIST`. Добавить в `config.py`:
```python
MAX_CONTEXT_DIST = 0.45  # блоки дальше этого порога не подмешиваются
```
В `sniper.py` → `generate_super_prompt()` — фильтровать результаты перед сборкой.

### 2. Убрать thoughts из контекста super-prompt

Проблема: при `THOUGHT_MODE=SMART` или `ON` в `cleaned_content` блока попадают куски размышлений модели (`"I'm currently focused on..."`). Они бесполезны для LLM, получающей промпт.

Решение: в `generate_super_prompt()` при сборке контекста — strip thoughts из текста блоков. Варианты:
- Хранить в metadata блока флаг `has_thoughts` и чистый `model_answer` отдельно.
- Или: при сборке промпта regex-ом убирать абзацы, начинающиеся с `**` и содержащие thought-паттерны (`"I'm currently"`, `"I'm now"`, `"I've been"`).
- Или: добавить поле `answer_only` в DialogBlock при парсинге.

### 3. Метаданные блоков в контексте промпта

Сейчас контекст — просто склеенные тексты через `---`. Нужно добавить заголовок к каждому блоку:
```
### Источник: {chat_id} (блок {chunk_index})
{cleaned_content без thoughts}
```
Это поможет LLM понять, откуда пришла информация.

### 4. Генерация .md файла на Google Drive

Добавить метод `ContextSniper.save_super_prompt()`:
```python
def save_super_prompt(self, main_prompt, meta_query=None, top_k=None, output_dir=None) -> Path:
    """Собирает промпт и сохраняет как .md файл."""
    prompt = self.generate_super_prompt(main_prompt, meta_query or main_prompt, top_k)
    output_dir = Path(output_dir or config.DB_DIR / "prompts")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%Y-%m-%d_%H%M}_{slugify(main_prompt[:40])}.md"
    path = output_dir / filename
    path.write_text(prompt, encoding="utf-8")
    return path
```
Если `meta_query` не передан — использовать `main_prompt` как запрос для поиска контекста.

### 5. Обновлённый тест для блока 3

После реализации пунктов 1–4 — прогнать обновлённый тест (`test_in_colab/test_for_b3_t1_dump.txt`) и сравнить:
- Thoughts исчезли из контекста
- Метаданные (chat_id) видны в промпте
- Нерелевантные блоки (dist > 0.45) не попадают в контекст
- .md файл сохраняется на Drive

---

## Tier 1.5 — после фикса super-prompt

### 6. Подключить LLM к таггеру через OpenRouter (~15 мин)

Провайдер: OpenRouter, модель: `deepseek/deepseek-v4-flash` ($0.14/1M input, $0.28/1M output).

Создать файл `core/providers/openrouter.py`:

```python
import requests

def openrouter_call(prompt: str, api_key: str, model: str = "deepseek/deepseek-v4-flash") -> str:
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

Затем в ноутбуке:
```python
from google.colab import userdata
from core.providers.openrouter import openrouter_call
from functools import partial

llm = partial(openrouter_call, api_key=userdata.get('OPENROUTER_API_KEY'))
sniper.tag_chats(llm, only_untagged=True)
```

API-ключ хранить в Colab Secrets как `OPENROUTER_API_KEY`.

### 7. Добавить tqdm в ингест (~5 мин)

В `sniper.py` → `ingest()`:
```python
from tqdm import tqdm
for path in tqdm(list(ingestion.iter_files(root)), desc="scanning"):
    ...
```

Добавить `tqdm` в `requirements.txt`.

### 8. Прогон на полном архиве

После фикса super-prompt и подключения таггера — прогнать на полном архиве чатов.

## Tier 2 — улучшения после стабилизации

### 9. Добавить harrier-oss-v1-0.6b в реестр моделей

Microsoft `harrier-oss-v1-0.6b` — топ MMTEB в классе 0.6B, мультиязычная, MIT, sentence-transformers совместимая. Добавить запись в `core/embeddings.py`:

```python
"harrier": EmbedderSpec(
    key="harrier",
    hf_name="microsoft/harrier-oss-v1-0.6b",
    dim=...,          # уточнить из карточки модели
    max_seq_len=...,  # уточнить
    query_prefix="",  # уточнить
    passage_prefix="",
),
```

После добавления — прогнать ингест с `config.EMBEDDING_MODEL = "harrier"` и сравнить качество поиска с bge-m3.

### 10. Зеркалирование тегов в Chroma

Добавить метод `ContextSniper.sync_tags_to_chroma()`:
- Пройтись по `manifest.chat_paths()`.
- Для каждого `chat_id` с непустыми `tags`/`topics` — обновить metadata всех блоков этого чата в Chroma через `collection.update(...)`.
- После этого `search_context(query, where={"tags": {"$contains": "#siem"}})` заработает.

### 11. CLI-обёртка (опционально, если есть время)

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
