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
DATASET_URL = "https://raw.githubusercontent.com/example/rusentiment.csv"  # ЗАМЕНИТЬ НА РЕАЛЬНУЮ ССЫЛКУ
DATASET_PATH = os.path.join(DATA_DIR, "rusentiment.csv")

# Параметры предобработки
MAX_WORDS = 10000        # Максимальный размер словаря
MAX_SEQUENCE_LENGTH = 150  # Максимальная длина текста в словах

# Параметры модели
NUM_CLASSES = 3          # Позитивный, Негативный, Нейтральный
EPOCHS = 10              # Количество эпох для тестов (потом увеличим)
BATCH_SIZE = 32