import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import aic
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # Чтобы игнорировать предупреждения об инвертируемых параметрах

# Загрузка данных
dataset_1 = pd.read_csv("1.csv", delimiter=";", decimal=",")
dataset_2 = pd.read_csv("1_2.csv", delimiter=";", decimal=",")

# Инициализация списка для хранения результатов
results = []

# Предполагаем, что вы найдете правильное название столбца и замените "your_column_name" на него
data_column_1 = dataset_1["data"]
data_column_2 = dataset_2["yuan"]

# Длины мерных интервалов, которые мы хотим исследовать
intervals = [30, 60, 90]

for interval in intervals:
    for p in range(3):  # Перебираем разные значения p
        for d in range(2):  # Перебираем разные значения d
            for q in range(3):  # Перебираем разные значения q
                try:
                    # Строим модель SARIMA
                    model_1 = SARIMAX(data_column_1, order=(p, d, q), seasonal_order=(0, 0, 0, interval))
                    model_fit_1 = model_1.fit()

                    model_2 = SARIMAX(data_column_2, order=(p, d, q), seasonal_order=(0, 0, 0, interval))
                    model_fit_2 = model_2.fit()

                    # Получаем AIC для каждой модели
                    aic_1 = model_fit_1.aic
                    aic_2 = model_fit_2.aic

                    # Сохраняем результаты
                    results.append({"Длина мерного интервала": interval,
                                    "p": p,
                                    "q": q,
                                    "AIC - data": aic_1,
                                    "AIC - yuan": aic_2})
                except:
                    continue

# Преобразуем результаты в DataFrame
results_df = pd.DataFrame(results)

# Для каждой длины мерного интервала выбираем модель с минимальным AIC
best_models = (
    results_df.loc[results_df.groupby("Длина мерного интервала")["AIC - data"].idxmin()].reset_index(drop=True))

# Выводим результаты
print(best_models)
