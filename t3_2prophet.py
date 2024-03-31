import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt


def prophet_model(df, period, test):
    # Создание экземпляра модели Prophet
    model = Prophet()

    # Добавление данных
    model.fit(df[:period])  # Обучаем модель на первых `period` точках

    # Создание будущего DataFrame для предсказания на оставшихся точках
    future = model.make_future_dataframe(periods=len(df) - period)

    # Предсказание значений
    forecast = model.predict(future)

    # Вычисление среднеквадратичного отклонения между фактическими и предсказанными значениями
    rmse = np.sqrt(np.mean((df['y'][period:period+test] - forecast['yhat'][period:period+test]) ** 2))

    print("RMSE:", round(rmse, 4))

    # Графики для визуализации
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'][:period+test], df['y'][:period+test], label='Фактические значения')
    plt.plot(forecast['ds'][:period+test], forecast['yhat'][:period+test], label='Предсказанные значения')
    plt.title("Прогноз на основе модели Prophet")
    plt.xlabel("Дата")
    plt.ylabel("Значение")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(df['ds'][:period+test], df['y'][:period+test] - forecast['yhat'][:period+test], label='Разность')
    plt.title("Разность между фактическими и предсказанными значениями")
    plt.xlabel("Дата")
    plt.ylabel("Разность")
    plt.legend()
    plt.show()


# -------------------------

data_x = pd.read_csv("data_x.csv", delimiter=";", decimal=",")
data_yuan = pd.read_csv("yuan.csv", delimiter=";", decimal=",")

# Извлекаем данные из столбцов
x = data_x["data"]
yuan = data_yuan["yuan"]

# Создание диапазона дат, т.к. нужен именно dataFrame с датами
date_range = pd.date_range(start='2024-01-01', periods=len(x), freq='D')
df = pd.DataFrame({'ds': date_range, 'y': np.array(x)})

# Для X
prophet_model(df, period=45, test=10)

# Для юань
# Создание диапазона дат, т.к. нужен именно dataFrame с датами
# Создание диапазона дат, т.к. нужен именно dataFrame с датами
date_range = pd.date_range(start='2023-11-01', periods=len(yuan), freq='D')

df = pd.DataFrame({'ds': date_range, 'y': np.array(yuan)})
prophet_model(df, period=250, test=20)
