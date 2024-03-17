import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt

# Загружаем данные
dataset_1 = pd.read_csv("1.csv", delimiter=";", decimal=",")
dataset_2 = pd.read_csv("1_2.csv", delimiter=";", decimal=",")

# Проверяем названия столбцов в DataFrame
print(dataset_1.head())
print(dataset_2.head())

# Предположим, что вы найдете правильное название столбца и замените "your_column_name" на него
data_column_1 = dataset_1["data"]
print()
data_column_2 = dataset_2["yuan"]

model_1 = ARIMA(data_column_1, seasonal_order=(2, 0, 0, 87))
model_fit_1 = model_1.fit()

model_2 = ARIMA(data_column_2, seasonal_order=(2, 0, 0, 276))
model_fit_2 = model_2.fit()

# Вывод результатов модели
print(model_fit_1.summary())
print(model_fit_2.summary())

# Получение предсказанных значений и доверительных интервалов
pred_1 = model_fit_1.get_prediction(start=0, end=len(data_column_1) - 1)
pred_2 = model_fit_2.get_prediction(start=0, end=len(data_column_2) - 1)

# Вычисление среднеквадратического отклонения по уровню доверительной вероятности 0.95
rmse_1 = rmse(data_column_1, pred_1.predicted_mean)
rmse_2 = rmse(data_column_2, pred_2.predicted_mean)

# Вывод результатов
print("Cреднеквадратическое отклонение для первого файла:", rmse_1)
print("Cреднеквадратическое отклонение для второго файла:", rmse_2)

# Прогнозирование будущих значений
forecast_1 = model_fit_1.forecast(steps=10)  # Пример: прогноз на 10 шагов вперёд
forecast_2 = model_fit_2.forecast(steps=10)  # Пример: прогноз на 10 шагов вперёд

print("Прогноз первого файла:", forecast_1)
print("Прогноз второго файла:", forecast_2)
