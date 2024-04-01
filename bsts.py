import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybuc import buc


def forecast_accuracy(actual, forecast):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual)) # MAPE
    rmse = np.mean((forecast - actual) ** 2) ** 0.5 # RMSE
    return {'mape': mape, 'rmse': rmse}


# Считывание данных из файла
file = "yuan1.csv"  # Укажите имя вашего файла с данными о курсе юаня
df = pd.read_csv(file, sep=';', header=None, names=['time', 'value'])
print(df)

# Преобразование данных в формат временного ряда
time_series = pd.Series(df['value'].values, index=pd.to_datetime(df['time'], format="%d.%m.%Y %H:%M"))

# Разделение данных на обучающую и тестовую выборки
train_data = time_series[:-270]  # Последние 100 значений будут использоваться для тестирования
test_data = time_series[-270:]

# Создание модели Байесовских временных рядов
bayes_uc = buc.BayesianUnobservedComponents(response=train_data, level=True, stochastic_level=True, trend=True,
                                            stochastic_trend=True, trig_seasonal=((2, 0),),
                                            stochastic_trig_seasonal=(True,))
post = bayes_uc.sample(5000)
#  метод sample, который используется для выполнения семплирования из постериорного распределения модели. Здесь 5000
#  - это количество сэмплов, которые мы хотим получить из постериорного распределения. Чем больше количество сэмплов,
#  тем более точные оценки мы можем получить, но это также увеличивает время выполнения.
mcmc_burn = 0
#  количество "прогревочных" итераций (burn-in iterations) для метода MCMC (Markov Chain Monte Carlo), которые мы хотим
#  использовать перед тем, как начать сохранять сэмплы из постериорного распределения. Здесь установлено значение 0, что
#  означает, что мы не будем использовать прогревочные итерации, и начнем сохранять сэмплы сразу после начала
#  сэмплирования.

# Получение и построение прогноза
forecast, _ = bayes_uc.forecast(100, mcmc_burn)

# Рассчитываем квантили 0.025 и 0.975 для интервала с уровнем доверия 0.95
# lower_quantile = np.percentile(forecast, 2.5, axis=0)
# upper_quantile = np.percentile(forecast, 97.5, axis=0)

forecast_mean = np.mean(forecast, axis=0)
plt.plot(test_data)
plt.plot(bayes_uc.future_time_index, forecast_mean)
plt.title('Прогноз курса юаня')
plt.legend(['Истинное значение', 'Прогноз'])
plt.show()

# Расчет и вывод MAPE и RMSE
accuracy = forecast_accuracy(test_data.values, forecast_mean)
print("Точность прогноза:")
print("MAPE:", accuracy['mape'])
print("RMSE:", accuracy['rmse'])

# Рассчитываем среднеквадратичное отклонение на уровне доверия 0.95
# std_95 = (upper_quantile - lower_quantile) / 3.92  # Коэффициент для уровня доверия 0.95
# print("Среднеквадратичное отклонение (0.95):", np.mean(std_95))
