"""
Модуль оценки качества модели.
Вычисляет метрики: accuracy, precision, recall, F1-score, confusion matrix.
Автор: Пастухова Елизавета Владимировна
Тема ВКР: Интеллектуальный анализ и прогноз тональности сообщений в социальных сетях
"""

import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from src.vectorization import load_tokenizer, texts_to_sequences
from src.preprocessing import preprocess_text
from src.config import MODELS_DIR, ID_TO_LABEL


def load_trained_model():
    """
    Загружает обученную модель и токенизатор.
    """
    model_path = os.path.join(MODELS_DIR, "sentiment_model.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    print(f" Загрузка модели из: {model_path}")
    model = load_model(model_path)
    tokenizer = load_tokenizer()

    print(" Модель и токенизатор загружены")
    return model, tokenizer


def predict_sentiment(model, tokenizer, text: str):
    """
    Предсказывает тональность одного текстового сообщения.
    Возвращает: (метка класса, вероятность, все вероятности)
    """
    # Предобрабатываем текст
    cleaned_text = preprocess_text(text)

    # Векторизуем
    sequence = texts_to_sequences(tokenizer, [cleaned_text])

    # Получаем предсказание
    predictions = model.predict(sequence, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    # Получаем метку класса (positive, negative, neutral)
    label = ID_TO_LABEL.get(predicted_class, f"Class_{predicted_class}")

    return label, confidence, predictions


def evaluate_model_detailed(model, X_test, y_test):
    """
    Проводит детальную оценку модели на тестовой выборке.
    Вычисляет: accuracy, precision, recall, F1, confusion matrix.
    """
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНАЯ ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    print("=" * 60)

    # Получаем предсказания
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Вычисляем метрики
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"])
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Выводим результаты
    print(f"\n Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\n Classification Report:")
    print(report)
    print(f" Confusion Matrix:")
    print(conf_matrix)

    # Сохраняем отчёт в файл
    save_evaluation_report(accuracy, report, conf_matrix)

    return accuracy, report, conf_matrix


def save_evaluation_report(accuracy, report, conf_matrix):
    """
    Сохраняет подробный отчёт об оценке в текстовый файл.
    """
    report_path = os.path.join(MODELS_DIR, "evaluation_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЁТ ОБ ОЦЕНКЕ КАЧЕСТВА МОДЕЛИ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + "\n")

    print(f"\n Отчёт сохранён: {report_path}")


def test_single_prediction(model, tokenizer, text: str):
    """
    Тестирует предсказание на одном примере.
    """
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ НА ОДНОМ ПРИМЕРЕ")
    print("=" * 60)

    print(f"\n Исходный текст: {text}")

    label, confidence, all_probs = predict_sentiment(model, tokenizer, text)

    print(f"\n Предсказанная тональность: {label}")
    print(f" Уверенность модели: {confidence:.4f} ({confidence * 100:.2f}%)")
    print(f"\n Все вероятности:")
    for i, prob in enumerate(all_probs):
        class_name = ID_TO_LABEL.get(i, f"Class_{i}")
        print(f"   {class_name}: {prob:.4f} ({prob * 100:.2f}%)")


def main():
    """
    Главная функция: загружает модель и проводит детальную оценку.
    """
    print("\n" + "🔍" * 30)
    print("ЗАПУСК ОЦЕНКИ КАЧЕСТВА МОДЕЛИ")
    print("🔍" * 30 + "\n")

    try:
        # Загружаем модель
        model, tokenizer = load_trained_model()

        # Тестируем на одном примере
        test_text = "Этот университет просто замечательный! Очень доволен обучением."
        test_single_prediction(model, tokenizer, test_text)

        print("\n Для полной оценки на тестовой выборке нужно запустить train.py")
        print("   (там автоматически вызывается evaluate_model_detailed)")

    except Exception as e:
        print(f"\n Ошибка при оценке модели: {e}")
        raise


if __name__ == "__main__":
    main()