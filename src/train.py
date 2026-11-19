"""
Модуль обучения нейронной сети.
Загружает данные, предобрабатывает их, обучает модель и сохраняет результаты.
Автор: Пастухова Елизавета Владимировна
Тема ВКР: Интеллектуальный анализ и прогноз тональности сообщений в социальных сетях
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from src.config import (
    DATASET_PATH, MODELS_DIR, EPOCHS, BATCH_SIZE,
    MAX_WORDS, MAX_SEQUENCE_LENGTH, NUM_CLASSES
)
from src.data_loader import download_dataset, load_and_inspect_data
from src.preprocessing import preprocess_text
from src.vectorization import build_tokenizer, texts_to_sequences, save_tokenizer
from src.model import build_model


def load_and_prepare_data():
    """
    Загружает датасет и подготавливает данные для обучения.
    Возвращает: X_train, X_test, y_train, y_test, tokenizer
    """
    print("=" * 60)
    print("ШАГ 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    print("=" * 60)

    # Проверяем, существует ли датасет
    if not os.path.exists(DATASET_PATH):
        print("📥 Датасет не найден. Начинаем загрузку...")
        if not download_dataset():
            raise FileNotFoundError("Не удалось загрузить датасет. Проверьте ссылку в config.py")

    # Загружаем данные
    print(f" Загрузка данных из: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f" Загружено строк: {len(df)}")

    # Проверяем наличие необходимых колонок
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError(f"Датасет должен содержать колонки 'text' и 'sentiment'. Найдены: {df.columns.tolist()}")

    # Удаляем строки с пропущенными значениями
    df = df.dropna(subset=['text', 'sentiment'])
    print(f" После удаления пропусков: {len(df)} строк")

    # Разделяем данные на текст и метки
    texts = df['text'].astype(str).tolist()
    labels = df['sentiment'].astype(int).tolist()

    # Предобрабатываем текст
    print(" Предобработка текста (это может занять время)...")
    cleaned_texts = [preprocess_text(text) for text in texts]
    print(f" Предобработка завершена")

    # Создаём и обучаем токенизатор
    tokenizer = build_tokenizer(cleaned_texts)

    # Преобразуем текст в числовые последовательности
    print(" Векторизация текста...")
    sequences = texts_to_sequences(tokenizer, cleaned_texts)
    print(f" Векторизация завершена. Shape: {sequences.shape}")

    # Разделяем на обучающую и тестовую выборки (80% / 20%)
    print(" Разделение данных на train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        sequences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # Сохраняем пропорции классов
    )
    print(f" Train: {len(X_train)} строк, Test: {len(X_test)} строк")

    return X_train, X_test, y_train, y_test, tokenizer


def train_model(X_train, y_train, X_test, y_test):
    """
    Обучает нейронную сеть на подготовленных данных.
    Возвращает: обученная модель, история обучения
    """
    print("\n" + "=" * 60)
    print("ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 60)

    # Создаём модель
    model = build_model()

    # Настраиваем раннюю остановку (чтобы не переобучаться)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Обучаем модель
    print(f" Начинаем обучение на {EPOCHS} эпохах...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    print(" Обучение завершено!")

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Оценивает качество модели на тестовой выборке.
    """
    print("\n" + "=" * 60)
    print("ШАГ 3: ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    print("=" * 60)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f" Результаты на тестовой выборке:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   Loss: {loss:.4f}")

    if accuracy >= 0.70:
        print(f" Целевой показатель Accuracy ≥ 70% ДОСТИГНУТ!")
    else:
        print(f"⚠ Целевой показатель Accuracy ≥ 70% НЕ ДОСТИГНУТ. Требуется улучшение модели.")

    return accuracy


def save_model(model, tokenizer, accuracy):
    """
    Сохраняет обученную модель и токенизатор.
    """
    print("\n" + "=" * 60)
    print("ШАГ 4: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "sentiment_model.h5")
    model.save(model_path)
    print(f" Модель сохранена: {model_path}")

    save_tokenizer(tokenizer)

    metrics_path = os.path.join(MODELS_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Target: 0.70\n")
        f.write(f"Achieved: {'Yes' if accuracy >= 0.70 else 'No'}\n")
    print(f" Метрики сохранены: {metrics_path}")


def main():
    """
    Главная функция: запускает полный пайплайн обучения.
    """
    print("\n" + "🚀" * 30)
    print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ АНАЛИЗА ТОНАЛЬНОСТИ")
    print("🚀" * 30 + "\n")

    try:
        X_train, X_test, y_train, y_test, tokenizer = load_and_prepare_data()
        model, history = train_model(X_train, y_train, X_test, y_test)
        accuracy = evaluate_model(model, X_test, y_test)
        save_model(model, tokenizer, accuracy)

        print("\n" + "🎉" * 30)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("🎉" * 30 + "\n")

    except Exception as e:
        print(f"\n Ошибка в процессе обучения: {e}")
        raise


if __name__ == "__main__":
    main()