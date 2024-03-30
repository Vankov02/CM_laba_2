# import numpy as np
# from scipy.optimize import curve_fit
#
# def fourier_analysis(x, y, num_terms=10):
#     # Период сезонности
#     period = 3  # Предполагаем, что данные имеют годовую сезонность
#
#     # Вычисление коэффициентов Фурье
#     coefficients = np.fft.fft(y)
#     coefficients /= len(y)
#
#     # Отбираем только первые num_terms гармонических составляющих
#     coefficients = coefficients[:num_terms]
#
#     # Определение функции для модели
#     def fourier_series(x, *params):
#         series = np.zeros_like(x, dtype=np.float64)
#         for i in range(0, len(params), 2):
#             n = i // 2 + 1
#             series += params[i] * np.cos(2 * np.pi * n * x / period) + params[i + 1] * np.sin(
#                 2 * np.pi * n * x / period)
#         return series
#
#     # Применение метода наименьших квадратов для оценки параметров
#     p0 = [0.0] * (2 * num_terms)
#     params, _ = curve_fit(fourier_series, x, y, p0=p0)
#
#     # Вывод коэффициентов
#     print("Коэффициенты гармонических составляющих:")
#     for i in range(num_terms):
#         print(f"Гармоника {i+1}: a{i+1} = {params[2*i]:.4f}, b{i+1} = {params[2*i+1]:.4f}")
#
#     # Создание модели
#     def model(x):
#         return fourier_series(x, *params)
#
#     return model


import numpy as np
import matplotlib.pyplot as plt

def fourier_analysis(x, y):
    # N - длина временного ряда
    N = len(x)

    # values - временной ряд (данные)
    values = y

    # Применяем преобразование Фурье к данным
    fft_vals = np.fft.fft(values)

    # Получаем абсолютные значения амплитуд (модулей) преобразования Фурье,
    # так как преобразование Фурье возвращает комплексные числа
    fft_abs = fft_vals.real

    # Создаем массив частот для отображения на графике
    freqs = np.fft.fftfreq(len(x))

    # Выводим график спектра Фурье
    plt.figure(figsize=(10, 5))
    plt.plot(freqs[:N // 2], fft_abs[:N // 2])  # Выводим только половину спектра (положительные частоты)
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.title('Спектр Фурье')
    plt.show()

    # Выбираем n самых больших пиков
    n = 10  # Примерное количество выбираемых пиков
    indices = np.argsort(np.abs(fft_vals))[-n:]  # Индексы n наибольших амплитуд

    # Выполняем обратное преобразование Фурье только для выбранных частот
    filtered_fft_vals = np.zeros_like(fft_vals)
    filtered_fft_vals[indices] = fft_vals[indices]
    filtered_time_series = np.fft.ifft(filtered_fft_vals)

    # Строим график отфильтрованного временного ряда
    plt.plot(filtered_time_series)
    plt.title('После обратного преобразования')
    plt.show()

    # Возвращаем отфильтрованный временной ряд
    return filtered_time_series