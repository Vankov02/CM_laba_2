import numpy as np
import statsmodels.api as sm

def extract_trend(model_fit):
    # Получение тренда из модели
    trend = model_fit.data.orig_endog - (model_fit.predict() - model_fit.resid)
    # Проверка на линейность
    p = np.polyfit(np.arange(len(trend)), trend, 1)
    if p[1] == 0:  # Если свободный член равен нулю, то тренд линейный
        trend_type = "линейный"
    elif p[0] == 0:  # Если коэффициент при x равен нулю, то тренд константный
        trend_type = "константный"
    else:
        trend_type = "другой"
    return trend, trend_type

def extract_seasonality(model_fit):
    # Получение сезонной составляющей из модели
    seasonal = model_fit.predict() - model_fit.resid
    return seasonal

def check_residuals_normality(model_fit):
    # Проверка нормальности остатков
    residuals = model_fit.resid
    _, p_value = sm.stats.normal_ad(residuals)
    return residuals, p_value

def check_model_decomposition(model_fit, trend, seasonal, residuals):
    # Проверка, что исходная модель = тренд + сезон + остатки
    original = model_fit.data.orig_endog
    model_sum = trend + seasonal + residuals
    decomposition_check = np.allclose(original, model_sum)
    return decomposition_check

def component_analysis(model_fit_for_x, model_fit_for_yuan):
    # Для модели model_fit_for_x
    trend_x, trend_type_x = extract_trend(model_fit_for_x)
    seasonality_x = extract_seasonality(model_fit_for_x)
    residuals_x, p_value_x = check_residuals_normality(model_fit_for_x)
    decomposition_check_x = check_model_decomposition(model_fit_for_x, trend_x, seasonality_x, residuals_x)

    # Для модели model_fit_for_yuan
    trend_yuan, trend_type_yuan = extract_trend(model_fit_for_yuan)
    seasonality_yuan = extract_seasonality(model_fit_for_yuan)
    residuals_yuan, p_value_yuan = check_residuals_normality(model_fit_for_yuan)
    decomposition_check_yuan = check_model_decomposition(model_fit_for_yuan, trend_yuan, seasonality_yuan, residuals_yuan)

    print("Трендовая составляющая для модели model_fit_for_x:", trend_type_x)
    print("Данные тренда для модели model_fit_for_x:", trend_x)
    print("Трендовая составляющая для модели model_fit_for_yuan:", trend_type_yuan)
    print("Данные тренда для модели model_fit_for_yuan:", trend_yuan)

    print("Сезонная составляющая для модели model_fit_for_x:", seasonality_x)
    print("Сезонная составляющая для модели model_fit_for_yuan:", seasonality_yuan)

    print("Данные остатков для модели model_fit_for_x:", residuals_x)
    print("Данные остатков для модели model_fit_for_yuan:", residuals_yuan)

    print("Нормальность остатков для модели model_fit_for_x:", p_value_x)
    print("Нормальность остатков для модели model_fit_for_yuan:", p_value_yuan)

    print("Проверка, что исходная модель = тренд + сезон + остатки для модели model_fit_for_x:", decomposition_check_x)
    print("Проверка, что исходная модель = тренд + сезон + остатки для модели model_fit_for_yuan:", decomposition_check_yuan)