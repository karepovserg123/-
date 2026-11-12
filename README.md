# - Мой первый проект
Задание для цифровой кафедры.
# 
Это репозиторий, созданный для выполнения учебного задания, буду очень надеяться что этого вполне хватит для демонстрации того что бы показать что я пытаюсь разбираться в работе.
"""Однослойный перцептрон: pure-Python, NumPy и scikit-learn реализации."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

RANDOM_SEED = 42


# ---------- Данные ----------

X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=float)

DATASETS = {
    "AND": (X, np.array([1, 0, 0, 0])),
    "OR":  (X, np.array([1, 1, 1, 0])),
    "NOT": (np.array([[1.0], [0.0]]), np.array([0, 1])),
}


# ---------- Общая логика ----------

def threshold(z: float) -> int:
    """Классический порог перцептрона: 1 при z >= 0, иначе 0."""
    return 1 if z >= 0 else 0


def evaluate(model, X: np.ndarray, y: np.ndarray, title: str) -> None:
    """Единый вывод результатов для всех реализаций."""
    print(f"\n{title}:")
    preds = model.predict(X)
    for x_i, y_true, y_pred in zip(X, y, preds):
        print(f"  Вход: {x_i}, Истинное: {y_true}, Предсказанное: {y_pred}")
