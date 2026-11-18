"""
Модуль построения нейронной сети.
Создает архитектуру модели глубокого обучения для классификации тональности текста.
Автор: Пастухова Елизавета Владимировна
Тема ВКР: Интеллектуальный анализ и прогноз тональности сообщений в социальных сетях
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.config import MAX_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, NUM_CLASSES, LEARNING_RATE


def build_model():
    """
    Строит и компилирует нейронную сеть для классификации тональности.

    Архитектура:
    1. Embedding - преобразует слова в векторы (обучаемые эмбеддинги)
    2. GlobalAveragePooling1D - усредняет векторы по всему тексту
    3. Dense + ReLU - полносвязный слой с функцией активации ReLU
    4. Dropout - регуляризация для предотвращения переобучения
    5. Dense + Softmax - выходной слой с вероятностями классов
    """
    print(" Построение архитектуры нейронной сети...")

    model = Sequential([
        # Слой 1: Embedding
        # Преобразует каждое слово (номер) в вектор размерности EMBEDDING_DIM
        Embedding(
            input_dim=MAX_WORDS,  # Размер словаря (10000 слов)
            output_dim=EMBEDDING_DIM,  # Размерность вектора слова (128)
            input_length=MAX_SEQUENCE_LENGTH  # Длина входной последовательности (150)
        ),

        # Слой 2: GlobalAveragePooling1D
        # Усредняет все векторы слов в один вектор фиксированной длины
        GlobalAveragePooling1D(),

        # Слой 3: Dense (полносвязный слой)
        # 64 нейрона с функцией активации ReLU (Rectified Linear Unit)
        Dense(64, activation='relu'),

        # Слой 4: Dropout
        # Случайно "отключает" 50% нейронов во время обучения, чтобы модель не переобучалась
        Dropout(0.5),

        # Слой 5: Выходной Dense
        # NUM_CLASSES нейронов (3 класса: позитив, негатив, нейтрал)
        # Softmax преобразует выходы в вероятности (сумма = 1)
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Компиляция модели
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),  # Оптимизатор Adam
        loss='sparse_categorical_crossentropy',  # Функция потерь для многоклассовой классификации
        metrics=['accuracy']  # Метрика для отслеживания
    )

    print(" Модель построена и скомпилирована.")
    print(f" Архитектура модели:\n{model.summary()}")

    return model