import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt

from CM_laba_2.task3 import component_analysis

# Интервал для анализа
period = 30

# Загружаем данные
data_x = pd.read_csv("data_x.csv", delimiter=";", decimal=",")
data_yuan = pd.read_csv("yuan.csv", delimiter=";", decimal=",")

# Проверяем названия столбцов в DataFrame
print(data_x.head())
print(data_yuan.head())

# Извлекаем данные из столбцов
x = data_x["data"]
yuan = data_yuan["yuan"]

# Строим модель ARIMA по данным x
model_for_x = ARIMA(x, seasonal_order=(2, 0, 0, period))
# Обучаем модель на данных x
model_fit_for_x = model_for_x.fit()

# Строим модель ARIMA по данным yuan
model_for_yuan = ARIMA(yuan, seasonal_order=(2, 0, 0, period))
# Обучаем модель на данных по юань
model_fit_for_yuan = model_for_yuan.fit()

# Вывод результатов модели
print(model_fit_for_x.summary())
print(model_fit_for_yuan.summary())

# Получение предсказанных значений и доверительных интервалов
pred_1 = model_fit_for_x.get_prediction(start=0, end=len(x) - 1)
pred_2 = model_fit_for_yuan.get_prediction(start=0, end=len(yuan) - 1)

# Вычисление среднеквадратического отклонения по уровню доверительной вероятности 0.95
rmse_x = rmse(x, pred_1.predicted_mean)
rmse_yuan = rmse(yuan, pred_2.predicted_mean)

# Вывод результатов
print("Cреднеквадратическое отклонение для данных X:", rmse_x)
print("Cреднеквадратическое отклонение для Юань:", rmse_yuan)

# Прогнозирование будущих значений
forecast_1 = model_fit_for_yuan.forecast(steps=10)  # Пример: прогноз на 10 шагов вперёд
forecast_2 = model_fit_for_yuan.forecast(steps=10)  # Пример: прогноз на 10 шагов вперёд

print("Прогноз для ряда X\n:", forecast_1)
print("Прогноз для Юань:\n", forecast_2)

# 3. ---------------------------------------
# Функция выделяет трендовую, сезонную и остаточную составляющие
component_analysis(model_fit_for_x, model_fit_for_yuan)
