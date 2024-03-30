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
t = []
for i in range(1, len(x) + 1):
    t.append(i)

best_fit_model, min_mse, best_model_name = model_fit(np.array(t), x.values)
print(f"Трендовая составляющая: {best_model_name}, МНК: {min_mse}")

# Остатки
resids = x.values - best_fit_model(np.array(t))

# Проверка нормальности
statistic, p_value = shapiro(resids)
print("Статистика теста:", statistic)
print("p-value:", p_value)

# Оценка результата
alpha = 0.05
if p_value > alpha:
    print("Остатки имеют нормальное распределение (не отвергаем нулевую гипотезу)")
else:
    print("Остатки не имеют нормальное распределение (отвергаем нулевую гипотезу)")

# ------- ДЛЯ Юань
print('--------------------------------------')
print('Для Юань')

t_yuan = []
for i in range(1, len(yuan) + 1):
    t_yuan.append(i)

# best_fit_model_yuan - тренд
best_fit_model_yuan, min_mse_yuan, best_model_name_yuan = model_fit(np.array(t_yuan), yuan.values)
print(f"Трендовая составляющая: {best_model_name_yuan}, МНК: {min_mse_yuan}")

# Остатки
resids_yuan = yuan.values - best_fit_model_yuan(np.array(t_yuan))

# Проверка нормальности
statistic, p_value = shapiro(resids_yuan)
print("Статистика теста:", statistic)
print("p-value:", p_value)

# Оценка результата
alpha = 0.05
if p_value > alpha:
    print("Остатки имеют нормальное распределение (не отвергаем нулевую гипотезу)")
else:
    print("Остатки не имеют нормальное распределение (отвергаем нулевую гипотезу)")

    # Функция, в которую подставляем значения, чтоб получить сезонную составляющую
    seasonal_component = fourier_analysis(t_yuan, resids_yuan).real

    resids2 = resids_yuan - seasonal_component

    # График остатков
    plt.subplot(2, 1, 1)  # 2 строки, 1 столбец, первый график
    plt.plot(t_yuan, resids_yuan)
    plt.title('Остатки')

    # График остатков без сезонной компоненты
    plt.subplot(2, 1, 2)  # 2 строки, 1 столбец, второй график
    plt.plot(t_yuan, resids2)
    plt.title('Остатки без сезонной компоненты')

    # Отображение обоих графиков
    plt.show()

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
