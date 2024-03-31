import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def fourier_analysis(x, y, predict_interval):
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
    plt.title('Сезонная составляющая')
    plt.show()

    # Получаем угловые частоты для выбранных пиков
    angular_freqs = 2 * np.pi * freqs[indices]

    # Создаем матрицу синусов и косинусов для выбранных угловых частот
    sin_matrix = np.sin(np.outer(x, angular_freqs))
    cos_matrix = np.cos(np.outer(x, angular_freqs))

    # Объединяем синусы и косинусы в одну матрицу признаков
    features = np.concatenate((sin_matrix, cos_matrix), axis=1)

    # Обучаем модель на сезонной составляющей с использованием гармоник
    model = LinearRegression()
    model.fit(features, filtered_time_series.real)  # Обучаем модель на вещественной части отфильтрованного временного ряда

    # Выбираем прогноз сезонной компоненты на N + predict_interval значений
    new_x = np.arange(N, N + predict_interval).reshape(-1, 1)
    angular_freqs = 2 * np.pi * np.fft.fftfreq(len(x))[np.argsort(np.abs(np.fft.fft(y)))[-10:]]
    sin_matrix = np.sin(np.outer(new_x, angular_freqs))
    cos_matrix = np.cos(np.outer(new_x, angular_freqs))
    features = np.concatenate((sin_matrix, cos_matrix), axis=1)
    predicted_seasonal_component = model.predict(features)

    # Возвращаем отфильтрованный временной ряд
    return filtered_time_series, predicted_seasonal_component
