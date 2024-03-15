import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

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


model_1 = ARIMA(data_column_1, order=(1, 1, 1))
model_fit_1 = model_1.fit()

model_2 = ARIMA(data_column_2, order=(1, 1, 1))
model_fit_2 = model_2.fit()

# Вывод результатов модели
print(model_fit_1.summary())
print(model_fit_2.summary())

# Прогнозирование будущих значений
# Замените 'steps' на количество шагов, на которые вы хотите сделать прогноз
forecast_1 = model_fit_1.forecast(steps=10)  # Пример: прогноз на 10 шагов вперёд
forecast_2 = model_fit_2.forecast(steps=10)  # Пример: прогноз на 10 шагов вперёд

print("Прогноз первого файла:", forecast_1)
print("Прогноз второго файла:", forecast_2)
