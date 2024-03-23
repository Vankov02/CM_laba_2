import numpy as np
from scipy.optimize import curve_fit

# Вычисление значения полинома степени n
def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

# Вычисление экспоненты. Принимает x, а, b, c - коэффициенты экспоненты.
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

# Вычисление полинома чебышева. Принимает x и коэффициенты полинома Чебышева.
def chebyshev_poly(x, *coeffs):
    return sum(c * np.polynomial.chebyshev.Chebyshev.basis(i)(x) for i, c in enumerate(coeffs))

def model_fit(x, y):
    # Многочленная регрессия до степени 5
    poly_degrees = range(1, 6)  # Степени полиномов
    models = []  # Список моделей
    mses = []    # Список среднеквадратичных ошибок

    best_model = None
    min_mse = float('inf')
    best_model_name = ""

    for degree in poly_degrees:
        try:
            p0 = [1.0] * (degree + 1)  # Начальное предположение для параметров
            popt, _ = curve_fit(polynomial, x, y, p0=p0, maxfev=10000)  # Подгонка модели
            model = lambda x: polynomial(x, *popt)  # Создание лямбда-функции для модели
            models.append(model)  # Добавление модели в список
            mse = np.mean((y - model(x))**2)  # Вычисление среднеквадратичной ошибки
            mses.append(mse)  # Добавление среднеквадратичной ошибки в список
            if mse < min_mse:
                best_model = model
                min_mse = mse
                best_model_name = f"Полином {degree}-й степени"
            print(f"Полином {degree}-й степени: {popt}, СКО: {mse}")  # Вывод коэффициентов и среднеквадратичной ошибки
        except RuntimeError as e:
            print(f"Полином {degree}-й степени не удалось подогнать: {str(e)}")  # В случае ошибки выводим сообщение

    # Экспоненциальная модель
    try:
        popt_exp, _ = curve_fit(exponential, x, y, p0=[1.0, 1.0, 1.0], maxfev=10000)  # Подгонка экспоненциальной модели
        model_exp = lambda x: exponential(x, *popt_exp)  # Создание лямбда-функции для экспоненциальной модели
        mse_exp = np.mean((y - model_exp(x))**2)  # Вычисление среднеквадратичной ошибки
        models.append(model_exp)  # Добавление экспоненциальной модели в список
        mses.append(mse_exp)  # Добавление среднеквадратичной ошибки в список
        if mse_exp < min_mse:
            best_model = model_exp
            min_mse = mse_exp
            best_model_name = "Экспоненциальная модель"
        print(f"Экспоненциальная модель: {popt_exp}, СКО: {mse_exp}")  # Вывод коэффициентов и среднеквадратичной ошибки
    except RuntimeError as e:
        print(f"Не удалось подогнать экспоненциальную модель: {str(e)}")  # В случае ошибки выводим сообщение

    # Полином Чебышева
    cheb_degrees = range(1, 6)  # Степени полиномов Чебышева

    for degree in cheb_degrees:
        try:
            p0 = [1.0] * (degree + 1)  # Начальное предположение для параметров
            popt_cheb, _ = curve_fit(chebyshev_poly, x, y, p0=p0, maxfev=10000)  # Подгонка полинома Чебышева
            model_cheb = lambda x: chebyshev_poly(x, *popt_cheb)  # Создание лямбда-функции для полинома Чебышева
            mse_cheb = np.mean((y - model_cheb(x))**2)  # Вычисление среднеквадратичной ошибки
            models.append(model_cheb)  # Добавление полинома Чебышева в список
            mses.append(mse_cheb)  # Добавление среднеквадратичной ошибки в список
            if mse_cheb < min_mse:
                best_model = model_cheb
                min_mse = mse_cheb
                best_model_name = f"Полином Чебышева {degree}-й степени"
            print(f"Многочлен Чебышева {degree}-й степени: {popt_cheb}, СКО: {mse_cheb}")  # Вывод коэффициентов и среднеквадратичной ошибки
        except RuntimeError as e:
            print(f"Многочлен Чебышева {degree}-й степени не удалось подогнать: {str(e)}")  # В случае ошибки выводим сообщение

    # Возвращаем лучшую модель, минимальное СКО, коэффициенты и название модели
    return best_model, min_mse, best_model_name
