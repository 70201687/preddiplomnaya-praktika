"""
Модуль веб-интерфейса на базе Streamlit.
Предоставляет пользователю возможность ввода текста и получения прогноза тональности.
Автор: Пастухова Елизавета Владимировна
Тема ВКР: Интеллектуальный анализ и прогноз тональности сообщений в социальных сетях
"""

import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Импортируем наши модули
from src.config import MODELS_DIR, ID_TO_LABEL
from src.preprocessing import preprocess_text
from src.vectorization import load_tokenizer, texts_to_sequences

# Настройка страницы
st.set_page_config(
    page_title="Анализ тональности сообщений",
    page_icon="🧠",
    layout="centered"
)


# Кэшируем загрузку модели и токенизатора, чтобы не загружать их при каждом действии
@st.cache_resource
def load_resources():
    model_path = os.path.join(MODELS_DIR, "sentiment_model.h5")
    tokenizer_path = os.path.join(MODELS_DIR, "tokenizer.pickle")

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error("❌ Модель или токенизатор не найдены! Сначала запустите `src/train.py` для обучения модели.")
        return None, None

    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer


# Загружаем ресурсы
model, tokenizer = load_resources()

# --- БОКОВАЯ ПАНЕЛЬ (САЙДБАР) ---
st.sidebar.title("🧠 Навигация")
page = st.sidebar.radio(
    "Выберите раздел:",
    ["🏠 Главная (Анализ)", "ℹ️ О проекте", "📚 Справка"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Разработчик: Пастухова Е.В.\nЧОУВО «МУ им. С.Ю. Витте»")

# --- ГЛАВНАЯ СТРАНИЦА: АНАЛИЗ ---
if page == "🏠 Главная (Анализ)":
    st.title("📝 Анализ тональности сообщения")
    st.write(
        "Введите текст сообщения из социальной сети, чтобы определить его эмоциональную окраску (позитивная, негативная или нейтральная).")

    # Поле ввода текста
    user_input = st.text_area(
        "Введите ваш текст здесь:",
        height=150,
        placeholder="Например: Этот университет просто замечательный! Очень доволен качеством обучения и преподавателями."
    )

    # Кнопка запуска анализа
    if st.button("🔍 Проанализировать тональность", type="primary"):
        if not user_input.strip():
            st.warning("⚠ Пожалуйста, введите текст для анализа.")
        elif model is None or tokenizer is None:
            st.error(" Модель не загружена. Проверьте наличие файлов в папке models/.")
        else:
            with st.spinner(" Анализируем текст..."):
                # 1. Предобработка
                cleaned_text = preprocess_text(user_input)

                # 2. Векторизация
                sequence = texts_to_sequences(tokenizer, [cleaned_text])

                # 3. Предсказание
                predictions = model.predict(sequence, verbose=0)[0]
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]

                # 4. Получение названия класса
                label = ID_TO_LABEL.get(predicted_class, "Неизвестно")

            # 5. Отображение результата
            st.success(" Анализ завершён!")

            # Красивое отображение результата
            if label == "positive":
                st.markdown(f"### 🟢 Тональность: **Позитивная**")
            elif label == "negative":
                st.markdown(f"### 🔴 Тональность: **Негативная**")
            else:
                st.markdown(f"### 🟡 Тональность: **Нейтральная**")

            st.progress(float(confidence))
            st.write(f"**Уверенность модели:** {confidence:.2%}")

            # Показываем, как модель "видит" текст после очистки
            with st.expander("👁️ Показать, как модель видит текст после предобработки"):
                st.code(cleaned_text)

# --- СТРАНИЦА: О ПРОЕКТЕ ---
elif page == " О проекте":
    st.title(" О проекте")
    st.write("""
    Данный интеллектуальный сервис разработан в рамках преддипломной практики 
    по направлению 09.03.03 «Прикладная информатика».

    **Цель:** Повышение эффективности процесса мониторинга репутации ЧОУВО «МУ им. С.Ю. Витте» 
    в социальных сетях за счёт применения модели машинного обучения.

    **Технологический стек:**
    - Python 3.12
    - TensorFlow / Keras (глубокое обучение)
    - Scikit-learn, NLTK, pymorphy2 (обработка данных)
    - Streamlit (веб-интерфейс)
    """)

# --- СТРАНИЦА: СПРАВКА ---
elif page == "📚 Справка":
    st.title("📚 Справка по использованию")
    st.write("""
    **Как пользоваться сервисом:**
    1. Перейдите на вкладку "Главная (Анализ)".
    2. Вставьте или введите текст сообщения в текстовое поле.
    3. Нажмите кнопку "Проанализировать тональность".
    4. Система автоматически очистит текст от шума (ссылок, спецсимволов) и выдаст результат.

    **Возможные результаты:**
    - 🟢 **Позитивная**: Текст выражает одобрение, радость или благодарность.
    - 🔴 **Негативная**: Текст содержит жалобы, критику или недовольство.
    - 🟡 **Нейтральная**: Текст носит информационный характер без яркой эмоциональной окраски.
    """)