import numpy as np
from scipy.optimize import curve_fit

def fourier_analysis(x, y, num_terms=10):
    # Период сезонности
    period = 3  # Предполагаем, что данные имеют годовую сезонность

    # Вычисление коэффициентов Фурье
    coefficients = np.fft.fft(y)
    coefficients /= len(y)

    # Отбираем только первые num_terms гармонических составляющих
    coefficients = coefficients[:num_terms]

    # Определение функции для модели
    def fourier_series(x, *params):
        series = np.zeros_like(x, dtype=np.float64)
        for i in range(0, len(params), 2):
            n = i // 2 + 1
            series += params[i] * np.cos(2 * np.pi * n * x / period) + params[i + 1] * np.sin(
                2 * np.pi * n * x / period)
        return series

    # Применение метода наименьших квадратов для оценки параметров
    p0 = [0.0] * (2 * num_terms)
    params, _ = curve_fit(fourier_series, x, y, p0=p0)

    # Вывод коэффициентов
    print("Коэффициенты гармонических составляющих:")
    for i in range(num_terms):
        print(f"Гармоника {i+1}: a{i+1} = {params[2*i]:.4f}, b{i+1} = {params[2*i+1]:.4f}")

    # Создание модели
    def model(x):
        return fourier_series(x, *params)

    return model