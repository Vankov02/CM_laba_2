import pandas as pd
import pystan
import numpy as np

# Загрузка данных
dataset_1 = pd.read_csv("data_x.csv", delimiter=";", decimal=",")
dataset_2 = pd.read_csv("yuan.csv", delimiter=";", decimal=",")

# Предполагаем, что вы найдете правильное название столбца и замените "your_column_name" на него
data_column_1 = dataset_1["data"]
data_column_2 = dataset_2["yuan"]


# Определение функции для вычисления среднеквадратического отклонения
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Определение функции для построения и оценки модели BSTS
def build_and_evaluate_bsts(data, interval_length):
    # Построение BSTS модели
    bsts_model_code = """
    data {
        int<lower=0> T; // Количество наблюдений
        vector[T] y;    // Временной ряд
    }
    parameters {
        real mu;                // Среднее
        real<lower=0> sigma;    // Стандартное отклонение
    }
    model {
        y ~ normal(mu, sigma);  // Модель нормального распределения
    }
    """
    # Подготовка данных
    data_dict = {'T': len(data), 'y': data.values}
    # Компиляция модели
    compiled_model = pystan.StanModel(model_code=bsts_model_code)
    # Вычисление параметров модели
    bsts_fit = compiled_model.sampling(data=data_dict)
    # Получение результатов моделирования
    bsts_results = bsts_fit.extract(permuted=True)
    # Вычисление среднеквадратического отклонения
    rmse_value = compute_rmse(data[-interval_length:], bsts_results['mu'][-interval_length:])
    return rmse_value


# Создание пустого DataFrame для хранения результатов
results = pd.DataFrame(columns=['Длина мерного интервала', 'p', 'q', 'Cреднеквадратическое отклонение'])

# Определение длин мерных интервалов
interval_lengths = [len(data_column_1), len(data_column_2)]

# Определение параметров p и q
parameters_to_explore = [
    {'p': 1, 'q': 0},
    {'p': 0, 'q': 1},
    # Другие комбинации параметров для исследования
]

# Выполнение моделирования для каждого временного ряда и параметров p, q
for interval_length in interval_lengths:
    for params in parameters_to_explore:
        p = params['p']
        q = params['q']
        # Выполнение моделирования BSTS
        rmse_value = build_and_evaluate_bsts(data_column_1, interval_length)  # Передаем первый временной ряд
        results = results.append({'Длина мерного интервала': interval_length, 'p': p, 'q': q,
                                  'Cреднеквадратическое отклонение': rmse_value}, ignore_index=True)

# Вывод результатов
print(results)
