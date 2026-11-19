"""
Модуль конфигурации проекта.
Здесь хранятся все глобальные настройки, пути и гиперпараметры.
Автор: Пастухова Елизавета Владимировна
Тема ВКР: Интеллектуальный анализ и прогноз тональности сообщений в социальных сетях
"""

import os

# Базовые пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Создаем папки, если их нет
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Параметры датасета
# ВАЖНО: Сюда нужно будет вставить прямую ссылку на CSV-файл с Kaggle или GitHub
DATASET_URL = "https://raw.githubusercontent.com/searayeah/russian-sentiment-emotion-datasets/main/RuSentiment/rusentiment_preselected_posts.csv"
DATASET_PATH = os.path.join(DATA_DIR, "rusentiment.csv")

# Параметры предобработки
MAX_WORDS = 10000        # Максимальный размер словаря
MAX_SEQUENCE_LENGTH = 150  # Максимальная длина текста в словах

# Параметры модели
NUM_CLASSES = 3          # Позитивный, Негативный, Нейтральный
EMBEDDING_DIM = 128      # Размерность векторного представления слова
LEARNING_RATE = 0.001    # Скорость обучения оптимизатора
EPOCHS = 10              # Количество эпох для тестов (потом увеличим)
BATCH_SIZE = 32

# Словарь для преобразования ID класса в название
ID_TO_LABEL = {
    0: "negative",   # Негативная тональность
    1: "neutral",    # Нейтральная тональность
    2: "positive"    # Позитивная тональность
}

LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}