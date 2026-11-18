"""
Модуль для загрузки и первичной проверки датасета.
Обеспечивает прямую загрузку данных по URL, как того требуют методические указания.
Автор: Пастухова Елизавета Владимировна
"""

import os
import pandas as pd
import requests
from src.config import DATASET_URL, DATASET_PATH


def download_dataset():
    """
    Скачивает датасет по прямой ссылке, если его еще нет в папке data.
    """
    if os.path.exists(DATASET_PATH):
        print(f" Датасет уже существует по пути: {DATASET_PATH}")
        return True

    print(f" Начинаю загрузку датасета по ссылке: {DATASET_URL}")
    try:
        response = requests.get(DATASET_URL)
        response.raise_for_status()  # Проверка на ошибки HTTP

        with open(DATASET_PATH, "wb") as f:
            f.write(response.content)

        print(f" Датасет успешно сохранен в: {DATASET_PATH}")
        return True

    except requests.exceptions.RequestException as e:
        print(f" Ошибка при загрузке датасета: {e}")
        print(" Совет: Проверьте правильность ссылки в src/config.py")
        return False


def load_and_inspect_data():
    """
    Загружает CSV в pandas DataFrame и выводит базовую информацию.
    """
    if not os.path.exists(DATASET_PATH):
        print(" Файл с данными не найден. Сначала запустите download_dataset()")
        return None

    try:
        df = pd.read_csv(DATASET_PATH)
        print("\n--- ИНФОРМАЦИЯ О ДАТАСЕТЕ ---")
        print(f"Всего строк: {len(df)}")
        print(f"Колонки: {list(df.columns)}")
        print("\nПервые 5 строк:")
        print(df.head())
        print("-----------------------------\n")
        return df
    except Exception as e:
        print(f" Ошибка при чтении CSV: {e}")
        return None