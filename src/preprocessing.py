"""
Модуль предобработки текста.
Очищает текст от мусора, приводит к нижнему регистру и удаляет стоп-слова.
Автор: Пастухова Елизавета Владимировна
Тема ВКР: Интеллектуальный анализ и прогноз тональности сообщений в социальных сетях
"""

import re
import nltk
from nltk.corpus import stopwords

# Скачиваем необходимые данные NLTK при первом запуске
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Загружаем русские стоп-слова
STOP_WORDS = set(stopwords.words('russian'))


def clean_text(text: str) -> str:
    """
    Базовая очистка текста: удаление ссылок, HTML-тегов, спецсимволов.
    :param text: исходный текст
    :return: очищенный текст в нижнем регистре
    """
    if not isinstance(text, str):
        return ""

    # Удаляем ссылки (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Удаляем HTML-теги
    text = re.sub(r'<.*?>', '', text)
    # Удаляем упоминания (@username) и хэштеги (#tag)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Оставляем только русские и английские буквы и пробелы
    text = re.sub(r'[^а-яА-Яa-zA-Z\s]', ' ', text)
    # Убираем лишние пробелы и переводим в нижний регистр
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text


def remove_stop_words(text: str) -> str:
    """
    Удаляет из текста частые слова, не несущие смысловой нагрузки (и, в, на и т.д.).
    :param text: очищенный текст
    :return: текст без стоп-слов
    """
    words = text.split()
    # Оставляем только слова, которых нет в списке стоп-слов и которые длиннее 2 букв
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    return " ".join(filtered_words)


def preprocess_text(text: str) -> str:
    """
    Главная функция: применяет все этапы очистки последовательно.
    :param text: исходный текст
    :return: полностью обработанный текст
    """
    cleaned = clean_text(text)
    final_text = remove_stop_words(cleaned)
    return final_text