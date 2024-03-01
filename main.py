import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA

# Загружаем данные
dataset_1 = pd.read_csv("1.csv", delimiter=";", decimal=",")
dataset_2 = pd.read_csv("1_2.csv", delimiter=";", decimal=",")

# Проверяем названия столбцов в DataFrame
print(dataset_1.head())
print(dataset_2.head())

# Предположим, что вы найдете правильное название столбца и замените "your_column_name" на него
data_column_1 = dataset_1["data_1"]
print()
data_column_2 = dataset_2["x_pole"]


def get_dickey_fuller_test(data):
    # Проверяем на стационарность с помощью ADF-теста
    adf_result = adfuller(data)
    print("Результаты расширенного теста Дикки-Фуллера")
    print(f"Статистика расширенного теста Дикки-Фуллера: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print("Критические значения:", adf_result[4])
    if adf_result[0] > adf_result[4]["5%"]:
        print("Есть единичные корни, ряд не стационарен")
    else:
        print("Единичных корней нет, ряд стационарен")


def get_kpss_test(data):
    # Проверяем на стационарность с помощью KPSS-теста
    kpss_result = kpss(data)
    print("\nРезультаты теста Квятковского-Филлипса-Шмидта-Шина:")
    print(f"Статистика теста Квятковского-Филлипса-Шмидта-Шина: {kpss_result[0]}")
    print(f"p-value: {kpss_result[1]}")
    print(f"Используемые лаги: {kpss_result[2]}")
    print("Критические значения:")
    for key, value in kpss_result[3].items():
        print(f"   {key}: {value}")

    # Проверяем гипотезу о стационарности
    if kpss_result[1] < 0.05:
        print("Нулевая гипотеза о стационарности отвергается. Ряд нестационарен.")
    else:
        print("Нулевая гипотеза о стационарности не отвергается. Ряд стационарен.")


get_dickey_fuller_test(data_column_1)
get_dickey_fuller_test(data_column_2)

print()

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
