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