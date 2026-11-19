"""
Модуль подготовки меток датасета RuSentiment.
Оригинальный датасет содержит 4 класса тональности: -1, 0, 1, 2.
Для задачи трёхклассовой классификации (negative, neutral, positive)
класс 2 (speech act) отфильтровывается, а метки перекодируются в 0, 1, 2.
Автор: Пастухова Елизавета Владимировна
"""

import pandas as pd


def filter_and_remap_labels(df: pd.DataFrame,
                            text_col: str = "text",
                            label_col: str = "sentiment") -> pd.DataFrame:
    """
    Оставляет только классы -1, 0, 1 и перекодирует их в 0, 1, 2.
    :param df: исходный DataFrame с колонками text и sentiment
    :return: очищенный DataFrame с колонками text и label (0/1/2)
    """
    # Оставляем только три нужных класса
    df = df[df[label_col].isin([-1, 0, 1])].copy()

    # Карта перекодировки: -1 -> 0 (negative), 0 -> 1 (neutral), 1 -> 2 (positive)
    mapping = {-1: 0, 0: 1, 1: 2}
    df["label"] = df[label_col].map(mapping)

    # Оставляем только нужные колонки
    df = df[[text_col, "label"]].rename(columns={text_col: "text"})

    print(f" После фильтрации 4-го класса осталось строк: {len(df)}")
    print(f" Распределение классов: {df['label'].value_counts().to_dict()}")

    return df.reset_index(drop=True)