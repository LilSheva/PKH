from pathlib import Path

# Пути (под Colab; локально переопредели при необходимости)
CHATS_DIR = Path("/content/drive/MyDrive/ai_studio_dumps")
DB_DIR = Path("/content/drive/MyDrive/pkh_chroma")
MANIFEST_PATH = DB_DIR / "manifest.json"

# Эмбеддинги: bge-m3 | qwen3 | e5-instruct
EMBEDDING_MODEL = "bge-m3"

# Размышления модели: OFF | ON | SMART
THOUGHT_MODE = "SMART"
THOUGHT_LONG_THRESHOLD = 1000
THOUGHT_SIM_THRESHOLD = 0.4

# Очистка
MAX_TOKEN_LEN = 100

# Дедупликация
ENABLE_SEMANTIC_DEDUP = False
SEMANTIC_DEDUP_THRESHOLD = 0.97

# Поиск
DEFAULT_TOP_K = 8

