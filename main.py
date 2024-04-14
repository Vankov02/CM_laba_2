import math

import numpy
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse
import matplotlib.pyplot as plt
from scipy.stats import shapiro

from CM_laba_2.t3_get_seasonal_component import fourier_analysis
from CM_laba_2.t3_get_trend_component import model_fit

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

# ------- ДЛЯ X
print('--------------------------------------')
print('Для исходных данных')

# Массив из чисел от 1 до 87. Т.к. x из pandas он не хавает
x_len = 30
x_predict = 10

t = []
for i in range(1, x_len + 1):
    t.append(i)

best_fit_model, min_mse, best_model_name = model_fit(np.array(t), x.values[:x_len])
print(f"Трендовая составляющая: {best_model_name}, МНК: {min_mse}")

for i in range(x_len, x_len + x_predict):
    t.append(i)

trend_x = best_fit_model(np.array(t[:x_len]))
predict_x = best_fit_model(np.array(t))

# Остатки
resids_x = x.values[:x_len] - trend_x

# Проверка нормальности
statistic, p_value = shapiro(resids_x)
print("Статистика теста:", statistic)
print("p-value:", p_value)

# Оценка результата
alpha = 0.05
if p_value > alpha:
    print("Остатки имеют нормальное распределение (не отвергаем нулевую гипотезу)")
else:
    print("Остатки не имеют нормальное распределение (отвергаем нулевую гипотезу)")

rmse_x = rmse(x.values[x_len:x_len + x_predict], predict_x[x_len:x_len + x_predict])
print('Среднеквадратическое отклонение: ', rmse_x)

plt.figure(figsize=(10, 6))
plt.plot(t, x.values[:x_len + x_predict], label='Фактические значения')
plt.plot(t, predict_x, label='Предсказанные значения')
plt.title("Тренд/сезон")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.show()

# ------- ДЛЯ Юань
print('--------------------------------------')
print('Для Юань')
start_index = 200
# Размер обучающей выборки
yuan_len = 250
# Размер интервала предсказаний
predict_interval = 10

# Инициализируем массив t
t_yuan = []
for i in range(start_index, yuan_len):
    t_yuan.append(i)

# best_fit_model_yuan - модель, описывающая тренд
best_fit_model_yuan, min_mse_yuan, best_model_name_yuan = model_fit(np.array(t_yuan), yuan.values[start_index:yuan_len])
print(f"Трендовая составляющая: {best_model_name_yuan}, МНК: {min_mse_yuan}")

# Получаем трендовую составляющую обучающей выборки
trend_yuan = best_fit_model_yuan(np.array(t_yuan))

# Получаем остатки обучающей выборки
resids_yuan = yuan.values[start_index:yuan_len] - trend_yuan

# Проверяем остатки на нормальность по тесту Шапиро
statistic, p_value = shapiro(resids_yuan)
print("Статистика теста:", statistic)
print("p-value:", p_value)

# Значения сезонной составляющей обучающей выборки
seasonal_yuan = []
# seasonal_yuan2 = []
# Инициализируем модель, описывающую сезонную составляющую
# Далее мы используем модель для прогноза
# Предсказание сезонной компоненты(остатки)
predicted_seasonal_component = None
# Предсказание сезонной компоненты(Остатки 2 = остатки1 - ещё раз выделили сезонную)
# predicted_seasonal_component2 = None

# Оценка результата
alpha = 0.05
if p_value > alpha:
    print("Остатки имеют нормальное распределение (не отвергаем нулевую гипотезу)")
else:
    print("Остатки не имеют нормальное распределение (отвергаем нулевую гипотезу)")

    # seasonal_component - значения сезонной составляющей обучающей выборки
    seasonal_component, predicted_seasonal_component = fourier_analysis(t_yuan, resids_yuan,
                                                                        predict_interval)
    # получаем значения сезонной компоненты
    seasonal_yuan = seasonal_component.real

    resids2 = resids_yuan - seasonal_component.real

    # seasonal_component2, predicted_seasonal_component2 = fourier_analysis(t_yuan[:yuan_len], resids2,
    #                                                                       predict_interval)
    # seasonal_yuan2 = seasonal_component2.real
    #
    # resids3 = resids2 - seasonal_component2

    # График остатков
    plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, первый график
    plt.plot(t_yuan, resids_yuan)
    plt.title('Остатки')

    # График остатков без сезонной компоненты
    plt.subplot(2, 1, 2)  # 2 строки, 1 столбец, второй график
    plt.plot(t_yuan, resids2)
    plt.title('Остатки без сезонной компоненты')

    # # Отображение обоих графиков
    # plt.show()
    #
    # # График остатков
    # plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, первый график
    # plt.plot(t_yuan[:yuan_len], resids_yuan)
    # plt.title('Остатки')
    #
    # # График остатков без сезонной компоненты
    # plt.subplot(2, 1, 2)  # 2 строки, 1 столбец, второй график
    # plt.plot(t_yuan[:yuan_len], resids3)
    # plt.title('Остатки без сезонной компоненты')
    #
    # # Отображение обоих графиков
    # plt.show()

    # Проверка нормальности с помощью теста Шапиро-Уилка
    statistic, p_value = shapiro(resids2)

    # Вывод результатов
    print("Статистика теста:", statistic)
    print("p-value:", p_value)

    # Оценка результата
    alpha = 0.05
    if p_value > alpha:
        print("Остатки без сезонной компоненты имеют нормальное распределение (не отвергаем нулевую гипотезу)")
    else:
        print("Остатки без сезонной компоненты не имеют нормальное распределение (отвергаем нулевую гипотезу)")

# Увеличиваем временной ряд на predict_interval значений
for i in range(yuan_len + 1, yuan_len + 1 + predict_interval):
    t_yuan.append(i)

predict_trend = best_fit_model_yuan(np.array(t_yuan))

# seasonal_yuan = np.concatenate((seasonal_yuan + seasonal_yuan2, predicted_seasonal_component + predicted_seasonal_component2))

# Если сезонная компонента есть, то делаем конкатенацию из массивов
# сезонной компоненты и предскзаанной сезонной компоненты
# Иначе - заполняем массив нулями, чтоб прога не ломалась
if (predicted_seasonal_component is None):
    seasonal_yuan = [0] * ((yuan_len - start_index) + predict_interval)
else:
    seasonal_yuan = np.concatenate((seasonal_yuan, predicted_seasonal_component))

rmse_x = rmse(yuan.values[yuan_len:yuan_len + predict_interval],
              (predict_trend + seasonal_yuan)[-predict_interval])
print('Среднеквадратическое отклонение: ', rmse_x)

lower_graph_bound = start_index
upper_graph_bound = yuan_len + predict_interval

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, первый subplot
plt.plot(t_yuan, yuan.values[start_index:yuan_len + predict_interval],
         label='Фактические значения')
plt.plot(t_yuan,
         (predict_trend + seasonal_yuan),
         label='Предсказанные значения')
plt.title("Тренд/сезон")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()

# Создаем второй subplot
plt.subplot(2, 1, 2)  # 2 строки, 1 столбец, второй subplot
plt.plot(t_yuan,
         yuan.values[start_index:yuan_len + predict_interval] - (predict_trend + seasonal_yuan),
         label='Отклонение')
plt.title("Отклонение")
plt.xlabel("t")
plt.ylabel("Отклонение")

plt.tight_layout()
plt.show()
