"""
Модуль векторизации текста.
Преобразует очищенный текст в числовые последовательности для нейронной сети.
Автор: Пастухова Елизавета Владимировна
Тема ВКР: Интеллектуальный анализ и прогноз тональности сообщений в социальных сетях
"""

import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.config import MAX_WORDS, MAX_SEQUENCE_LENGTH, MODELS_DIR

# Путь для сохранения обученного токенизатора
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pickle")


def build_tokenizer(texts):
    """
    Создает и обучает токенизатор на переданных текстах.
    :param texts: список очищенных текстовых строк
    :return: обученный объект Tokenizer
    """
    print(" Создание и обучение токенизатора...")
    # num_words=MAX_WORDS означает, что мы возьмем только MAX_WORDS самых частых слов
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    print(f" Токенизатор создан. Размер словаря: {len(tokenizer.word_index) + 1}")
    return tokenizer


def texts_to_sequences(tokenizer, texts):
    """
    Преобразует список текстов в последовательности чисел и выравнивает их длину.
    :param tokenizer: обученный объект Tokenizer
    :param texts: список очищенных текстовых строк
    :return: матрица числовых последовательностей (numpy array)
    """
    # Превращаем слова в числа
    sequences = tokenizer.texts_to_sequences(texts)

    # Выравниваем длину: обрезаем длинные тексты и дополняем короткие нулями
    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",  # добавляем нули в конец
        truncating="post"  # обрезаем с конца, если текст слишком длинный
    )
    return padded_sequences


def save_tokenizer(tokenizer, filepath=TOKENIZER_PATH):
    """
    Сохраняет токенизатор в файл с помощью pickle, чтобы использовать его при предсказаниях.
    """
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f" Токенизатор успешно сохранен в: {filepath}")


def load_tokenizer(filepath=TOKENIZER_PATH):
    """
    Загружает ранее сохраненный токенизатор из файла.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл токенизатора не найден по пути: {filepath}")

    with open(filepath, "rb") as f:
        tokenizer = pickle.load(f)
    print(f" Токенизатор успешно загружен из: {filepath}")
    return tokenizer