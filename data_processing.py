import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime as dt
from datetime import date as dt_date
from datetime import timedelta, datetime, date
from scipy.stats import norm
from typing import List, Dict, Callable
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def get_option_data(ticker: str) -> pd.DataFrame:
    """
    Получает данные по опционам для заданного тикера и возвращает DataFrame.

    Параметры:
    ----------
    ticker : str
        Тикер актива (например, 'SPY').

    Возвращает:
    -----------
    pd.DataFrame
        DataFrame с данными по опционам, содержащий столбцы: STRIKE, C_IV, expiration_date, Tenor.
    """
    
    # Получаем данные по тикеру
    options_data = yf.Ticker(ticker)

    # Получаем все даты экспирации
    expiration_dates = options_data.options

    # Текущая дата для расчета срока до погашения
    today = pd.to_datetime(dt.today())

    # Собираем данные по опционам и рассчитываем Tenor
    data = pd.concat(
        [
            pd.DataFrame({
                'STRIKE': options_data.option_chain(exp_date).calls['strike'],
                'C_IV': options_data.option_chain(exp_date).calls['impliedVolatility'],
                'expiration_date': pd.to_datetime(exp_date)
            }).assign(Tenor=(pd.to_datetime(exp_date) - today).days / 365)
            for exp_date in expiration_dates
        ],
        ignore_index=True
    )

    # Фильтруем данные по сроку до погашения <= 1 год
    data = data[data['Tenor'] <= 1]

    return data

def black_scholes_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    Вычисляет цену колл-опциона по формуле Блэка-Шоулза.

    Параметры:
    S (float): Текущая цена базового актива (например, акции).
    K (float): Исполнительная цена (страйк) опциона.
    r (float): Безрисковая процентная ставка (в виде десятичной дроби).
    T (float): Время до истечения опциона (в годах).
    sigma (float): Вмененная волатильность базового актива (в виде десятичной дроби).

    Возвращает:
    float: Цена колл-опциона.
    """
    
    # Вычисление d1 и d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Вычисление цены колл-опциона
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

def create_ordering_constraints(size: int = 11) -> List[Dict[str, Callable]]:
    """
    Создает список ограничений для оптимизационной задачи, обеспечивающий порядок элементов в матрице.

    Параметры:
    size (int): Размер матрицы (по умолчанию 11).

    Возвращает:
    List[Dict[str, Callable]]: Список ограничений для оптимизации, включая неравенства и равенства.
    """
    
    ordering_constraints = []

    for i in range(size):
        for j in range(size):
            if j < i:
                # Условие для элементов до диагонали: возрастание
                ordering_constraints.append({
                    'type': 'ineq',
                    'fun': lambda p_values, i=i, j=j, size=size: p_values[size * i + j] - p_values[size * i + j - 1]
                })
            elif j > i:
                # Условие для элементов после диагонали: убывание
                ordering_constraints.append({
                    'type': 'ineq',
                    'fun': lambda p_values, i=i, j=j, size=size: p_values[size * i + j] - p_values[size * i + j + 1]
                })
        
        # Ограничение для строгого максимального значения на диагонали (максимум в каждой строке)
        max_index = size * i + i
        row_indices = [size * i + k for k in range(size) if k != i]
        ordering_constraints.append({
            'type': 'eq',
            'fun': lambda p_values, max_index=max_index, row_indices=row_indices: p_values[max_index] - max(p_values[row_indices]) - 1e-8
        })
    
    return ordering_constraints

def distribution_from_markov_chain(markov_chain_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет приближенную плотность вероятности для стационарного распределения 
    Марковской цепи на основе заданной матрицы переходов.

    Параметры:
    ----------
    markov_chain_matrix : pd.DataFrame
        Квадратная матрица переходов Марковской цепи, где строки и столбцы 
        соответствуют состояниям, а элементы обозначают вероятности переходов 
        между состояниями.

    Возвращает:
    -----------
    tuple[np.ndarray, np.ndarray]
        pdf_values : np.ndarray
            Значения плотности вероятности (после интерполяции и нормировки).
        continuous_states : np.ndarray
            Массив точек состояний, соответствующих значениям плотности вероятности.
    """
    
    # Получение состояний
    states = markov_chain_matrix.index.to_numpy()

    # Формирование системы уравнений для вычисления стационарного распределения
    A = np.transpose(markov_chain_matrix.values) - np.eye(markov_chain_matrix.shape[0])
    A = np.vstack([A, np.ones(markov_chain_matrix.shape[0])])  # Добавляем строку для условия суммы
    b = np.zeros(markov_chain_matrix.shape[0] + 1)
    b[-1] = 1  # Условие, что сумма π = 1

    # Решаем систему уравнений πP = π и сумма π = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    # Создаем интерполяционную функцию для стационарного распределения
    interp_func = interp1d(states, pi, kind='cubic', fill_value='extrapolate')

    # Генерируем непрерывные точки для оценки плотности вероятности
    continuous_states = np.linspace(states.min(), states.max(), 500)
    pdf_values = interp_func(continuous_states)

    # Устанавливаем значения PDF не меньше 0
    pdf_values = np.maximum(pdf_values, 0)

    # Нормировка функции для получения корректной плотности
    integral, _ = quad(interp_func, continuous_states.min(), continuous_states.max())
    pdf_values = pdf_values / integral

    return pdf_values, continuous_states

def recovery_iterations(data, instrument):
    
    selected_date = pd.to_datetime(dt_date.today())
    undelying_last = (yf.download(instrument, start=(pd.to_datetime(dt_date.today()) - timedelta(days=100)), progress=False)['Adj Close'])[-1]

    dates = []
    risk_neutral_std = []
    risk_neutral_mean = []
    risk_neutral_skewness = []
    risk_neutral_kurtosis = []
    recovered_std = []
    recovered_mean = []
    recovered_skewness = []
    recovered_kurtosis = []
    risk_neutral_15_perc_drop = []
    risk_neutral_20_perc_drop = []
    risk_neutral_25_perc_drop = []
    recovered_15_perc_drop = []
    recovered_20_perc_drop = []
    recovered_25_perc_drop = []

    C_IV_Matrix = data[['STRIKE', 'C_IV', 'Tenor']].copy()
    C_IV_Matrix['STRIKE'] = pd.to_numeric(C_IV_Matrix['STRIKE'], errors='coerce')
    C_IV_Matrix['C_IV'] = pd.to_numeric(C_IV_Matrix['C_IV'], errors='coerce')
    # Создаем сводную таблицу (pivot table), используя цены колл-опционов как значения
    C_IV_Matrix = C_IV_Matrix.pivot_table(values='C_IV', index='STRIKE', columns='Tenor').dropna(inplace=False)
    Tenor = [round((i + 1) / 12, 4) for i in range(12)]
    # Определяем процентные изменения относительно центрального значения Strike
    percent_changes = np.arange(-30, 35, 5) / 100  # От -30% до +30% с шагом 5%
    Strike = undelying_last * (1 + percent_changes)
    vols_surface_empty = pd.DataFrame(index=Strike, columns=Tenor)

    ##################################### Интерполяция #####################################
    X = C_IV_Matrix.columns.astype(float)
    Y = C_IV_Matrix.index.astype(float)
    Z = C_IV_Matrix.values
    interp_func = RectBivariateSpline(Y, X, Z, kx=1, ky=1)
    empty_matrix_tenor = vols_surface_empty.columns.astype(float)
    empty_matrix_strike = vols_surface_empty.index.astype(float)
    interpolated_values = interp_func(empty_matrix_strike, empty_matrix_tenor, grid=True)
    vols_surface_interpolated = pd.DataFrame(interpolated_values, index=vols_surface_empty.index, columns=vols_surface_empty.columns)
    vols_surface_interpolated.index.name = 'Strike'
    vols_surface_interpolated.columns.name = 'Tenor'

    ##################################### Расчет цен опционов #####################################
    # Расчет цен Call опционов с интерполированной implied volatility

    # Процентная ставка теперь равна загруженной процентной ставке
    ticker = "^TNX"  # Тикер для 10-летних государственных облигаций
    start_date = selected_date - timedelta(days=7)
    end_date = selected_date
    interest_rate_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    r = interest_rate_data['Adj Close'].iloc[-1] / 100

    call_prices = pd.DataFrame(index=vols_surface_interpolated.index, columns=vols_surface_interpolated.columns)
    for tenor in vols_surface_interpolated.columns:
        for strike in vols_surface_interpolated.index:
            volatility = vols_surface_interpolated.loc[strike, tenor]
            call_prices.loc[strike, tenor] = black_scholes_call(undelying_last, strike, r, tenor, volatility)

    ##################################### State price matrix #####################################
    # Матрица state price из цен колл опционов по формуле Breeden-Litzenberg
    state_prices = pd.DataFrame(index=call_prices.index, columns=call_prices.columns)
    r = interest_rate_data['Adj Close'].iloc[-1] / 100

    for i in range(1, len(call_prices.index) - 1):
        for j in range(len(call_prices.columns)):
            state_prices.iloc[i, j] = np.exp(r * call_prices.columns[i]) * (
                call_prices.iloc[i + 1, j] - 2 * call_prices.iloc[i, j] + call_prices.iloc[i - 1, j]) / (
                    (call_prices.index[i+1] - call_prices.index[i])**2)

    state_prices = state_prices.iloc[1:-1].copy()  # Используйте метод .copy() для явного создания копии DataFrame
    state_prices[state_prices < 0] = 0

    # Масштабирование первого столбца
    first_column_sum = state_prices.iloc[:, 0].sum()
    state_prices.iloc[:, 0] = state_prices.iloc[:, 0] / first_column_sum

    central_row = state_prices.index[len(state_prices) // 2]
    percent_change = (state_prices.index - central_row) / central_row
    percent_change = pd.Series(percent_change).round(2)
    state_prices.index = percent_change

    ##################################### Transition state prices matrix #####################################
    state_prices_values = state_prices.values
    
    def error_function(p_values): # Функция ошибок
        total_error = 0
        p_matrix = np.array(p_values).reshape((11, 11))
        for i in range(11):
            for j in range(1, 12):
                state_prices_hat_ij = np.dot(state_prices_values[i, 1:], p_matrix[:, i])
                total_error += (state_prices_hat_ij - state_prices_values[i][j - 1]) ** 2
        return total_error

    # Ограничения для значений элементов матрицы P (от 0 до 1)
    bounds = [(0, 1) for _ in range(11 * 11)]

    # Установка значений в центральную строку новой матрицы
    central_column_values = [row[0] for row in state_prices_values]
    central_row_index = int(len(state_prices_values) / 2)

    # Ограничение для суммы элементов в каждой строке
    row_sum_constraints = {'type': 'eq', 'fun': lambda p_values: np.array([p_values[i:i+11].sum() - 1 for i in range(0, 121, 11)])}

    # Используем "eq" для строгих ограничений на максимум по диагонали
    constraints = [row_sum_constraints] + create_ordering_constraints()

    # Создание случайных начальных значений, сумма элементов в каждой строке равна 1
    initial_guess = np.full((11, 11), 1 / 11)  # Задаем начальные значения равные 1/11 для каждого элемента
    initial_guess_flat = initial_guess.flatten()  # Преобразуем матрицу в одномерный массив

    # Минимизация функции ошибок с ограничениями и начальными значениями
    result = minimize(error_function, initial_guess_flat, bounds=bounds, constraints=constraints, method='SLSQP')

    P_matrix = result.x.reshape((11, 11))
    P_matrix[central_row_index] = central_column_values
    P_matrix = pd.DataFrame(P_matrix, index=state_prices.index, columns=state_prices.index)
    P_matrix = P_matrix.rename_axis(index='', columns='')
    P_matrix[P_matrix < 0] = 0

    #################################### Моменты риск-нейтральных вероятностей #####################################
    # Объект с датами
    dates.append(selected_date)

    pdf_rn, continious_states_rn = distribution_from_markov_chain(P_matrix)

    # Среднее значение
    rn_mean = np.trapz(continious_states_rn * pdf_rn, continious_states_rn)
    risk_neutral_mean.append(rn_mean)

    # Дисперсия
    variance = np.trapz((continious_states_rn - rn_mean) ** 2 * pdf_rn, continious_states_rn)
    rn_std = np.sqrt(variance)
    risk_neutral_std.append(rn_std)

    # Асимметрия
    rn_skewness = np.trapz((continious_states_rn - rn_mean) ** 3 * pdf_rn, continious_states_rn)
    rn_skewness /= rn_std ** 3  # Нормируем на стандартное отклонение
    risk_neutral_skewness.append(rn_skewness)

    # Эксцесс
    rn_kurtosis = np.trapz((continious_states_rn - rn_mean) ** 4 * pdf_rn, continious_states_rn)
    rn_kurtosis /= rn_std ** 4  # Нормируем на стандартное отклонение
    rn_kurtosis -= 3  # Вырезаем 3 для получения эксцесса
    risk_neutral_kurtosis.append(rn_kurtosis)

    # Вероятность упасть на 15%
    risk_neutral_15_perc_drop.append(P_matrix.loc[0, -0.15])
    # Вероятность упасть на 20%
    risk_neutral_20_perc_drop.append(P_matrix.loc[0, -0.20])
    # Вероятность упасть на 25%
    risk_neutral_25_perc_drop.append(P_matrix.loc[0, -0.25])

    ##################################### Ross recovery #####################################

    # Рассчитываем характеристический вектор z и характеристический корень λ
    e = np.ones((11, 1))  # Вектор из единиц
    D_inv = np.diag(1 / np.diag(P_matrix))  # Обратная диагональная матрица D
    z = np.linalg.solve(D_inv, e)
    lambda_value = np.dot(z.T, np.dot(P_matrix, z)) / np.dot(z.T, z)

    # Рассчитываем ядро ценообразования φ
    phi = z / z[0]

    # Рассчитываем матрицу истинных вероятностей F
    F_matrix = (1 / lambda_value) * np.dot(np.dot(P_matrix, D_inv), P_matrix)

    F_matrix = pd.DataFrame(F_matrix, index=state_prices.index, columns=state_prices.index)
    F_matrix = F_matrix.rename_axis(index='', columns='')
    F_matrix[F_matrix < 0] = 0
    F_matrix = F_matrix.div(F_matrix.sum(axis=1), axis=0)

    ##################################### Восстановленные моменты #####################################

    pdf_recovered, continious_states_recovered = distribution_from_markov_chain(F_matrix)

    # Среднее значение
    mean = np.trapz(continious_states_recovered * pdf_recovered, continious_states_recovered)
    recovered_mean.append(mean)

    # Дисперсия
    variance = np.trapz((continious_states_recovered - mean) ** 2 * pdf_recovered, continious_states_recovered)
    standard_deviation = np.sqrt(variance)
    recovered_std.append(standard_deviation)

    # Асимметрия
    skewness = np.trapz((continious_states_recovered - mean) ** 3 * pdf_recovered, continious_states_recovered)
    skewness /= standard_deviation ** 3  # Нормируем на стандартное отклонение
    recovered_skewness.append(skewness)

    # Эксцесс
    kurtosis = np.trapz((continious_states_recovered - mean) ** 4 * pdf_recovered, continious_states_recovered)
    kurtosis /= standard_deviation ** 4  # Нормируем на стандартное отклонение
    kurtosis -= 3  # Вырезаем 3 для получения эксцесса
    recovered_kurtosis.append(kurtosis)

    # Вероятность упасть на 15%
    recovered_15_perc_drop.append(F_matrix.loc[0, -0.15])
    # Вероятность упасть на 20%
    recovered_20_perc_drop.append(F_matrix.loc[0, -0.20])
    # Вероятность упасть на 25%
    recovered_25_perc_drop.append(F_matrix.loc[0, -0.25])

    recovered_data = {
        'risk_neutral_mean': risk_neutral_mean,
        'risk_neutral_std': risk_neutral_std,
        'risk_neutral_skewness': risk_neutral_skewness,
        'risk_neutral_kurtosis': risk_neutral_kurtosis,
        'risk_neutral_15_perc_drop': risk_neutral_15_perc_drop,
        'risk_neutral_20_perc_drop': risk_neutral_20_perc_drop,
        'risk_neutral_25_perc_drop': risk_neutral_25_perc_drop,
        'recovered_mean': recovered_mean,
        'recovered_std': recovered_std,
        'recovered_skewness': recovered_skewness,
        'recovered_kurtosis': recovered_kurtosis,
        'recovered_15_perc_drop': recovered_15_perc_drop,
        'recovered_20_perc_drop': recovered_20_perc_drop,
        'recovered_25_perc_drop': recovered_25_perc_drop,
    }

    recovered_data = pd.DataFrame(recovered_data, index=dates)

    return recovered_data, P_matrix, F_matrix

def RV_calc(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Вычисляет реализованную волатильность на основе логарифмических доходностей.

    Параметры:
    data (pd.DataFrame): Датафрейм, содержащий столбец 'Adj Close' с ценами закрытия.
    window (int): Размер окна для расчета скользящей средней (количество периодов).

    Возвращает:
    pd.Series: Серия реализованной волатильности, рассчитанная по заданному окну.
    """
    
    # Расчет логарифмических доходностей
    vols = pd.DataFrame()
    vols['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    
    # Расчет квадратов доходностей
    vols['Squared Returns'] = vols['Log Returns'] ** 2
    
    # Вычисление реализованной волатильности с использованием скользящей средней
    vols['Realized_Volatility_t'] = vols['Squared Returns'].rolling(window=window).mean().apply(np.sqrt)
    
    return vols['Realized_Volatility_t']

def plot_density(continious_states_true, pdf_true, continious_states_rn, pdf_rn):
    """
    Визуализирует физическую и риск-нейтральную плотности распределения.

    Параметры:
    continious_states_true (array-like): Набор состояний для физического распределения.
    pdf_true (array-like): Плотность физического распределения.
    continious_states_rn (array-like): Набор состояний для риск-нейтрального распределения.
    pdf_rn (array-like): Плотность риск-нейтрального распределения.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(continious_states_true, pdf_true, color='skyblue', linewidth=2, label='Физическая PDF')
    plt.plot(continious_states_rn, pdf_rn, color='orange', linewidth=2, label='Риск-нейтральная PDF')
    plt.xlabel('Ожидаемая дох-ть')
    plt.ylabel('Плотность')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    plt.grid(False)
    plt.show()

def plot_cdf(continious_states_true, cdf_true, continious_states_rn, cdf_rn):
    """
    Визуализирует физическую и риск-нейтральную кумулятивные функции распределения (CDF).

    Параметры:
    continious_states_true (array-like): Набор состояний для физического распределения.
    cdf_true (array-like): Кумулятивная функция физического распределения.
    continious_states_rn (array-like): Набор состояний для риск-нейтрального распределения.
    cdf_rn (array-like): Кумулятивная функция риск-нейтрального распределения.
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Создаем фигуру и оси
    ax.plot(continious_states_true, cdf_true, color='skyblue', linewidth=2, label='Физическая CDF')
    ax.plot(continious_states_rn, cdf_rn, color='orange', linewidth=2, label='Риск-нейтральная CDF')
    ax.set_xlabel('Ожидаемая доходность')
    ax.set_ylabel('Кумулятивная вероятность')
    ax.legend()
    ax.grid(False)
    
    return fig  # Возвращаем фигуру

def plot_volatility_forecast(realized_volatility_df, predictions, step=5):
    """
    Визуализирует прогноз реализованной волатильности на основе существующих данных и предсказанных точек.

    Параметры:
    realized_volatility_df (pd.DataFrame): Датафрейм с реализованной волатильностью. Ожидается столбец 'Realized_Volatility_t'.
    predictions (list или pd.Series): Прогнозные значения волатильности.
    step (int): Шаг прогноза в днях между предсказанными точками. По умолчанию 5.
    """
    # Получение столбца с реализованной волатильностью
    realized_volatility = realized_volatility_df['Realized_Volatility_t']
    
    # Определение последнего индекса и расчет будущих индексов с заданным шагом
    last_index = realized_volatility.index[-1]
    future_indices = [last_index + pd.DateOffset(days=i * step) for i in range(1, len(predictions) + 1)]
    
    # Построение графика
    fig, ax = plt.subplots(figsize=(12, 6))  # Создаем фигуру и оси
    ax.plot(realized_volatility, label='Realized Volatility', color='skyblue', linewidth=2)
    
    # Добавляем предсказанные точки
    ax.scatter(future_indices, predictions, color='red', label='Predicted Volatility', zorder=10)
    
    # Соединяем конец графика с прогнозными точками одной линией
    x_values = [last_index] + future_indices
    y_values = [realized_volatility.iloc[-1]] + predictions.tolist()
    
    # Проводим линию между последним значением и всеми прогнозами
    ax.plot(x_values, y_values, color='orange', linestyle='-', linewidth=2, label='Forecast Line')
    
    # Обозначаем точки на графике
    for i, (index, pred) in enumerate(zip(future_indices, predictions), start=1):
        ax.text(index, pred, f't+{i * step}', fontsize=10, color='red', ha='right')
    
    # Добавляем подписи и легенду
    ax.set_title("Realized Volatility Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Realized Volatility")
    ax.legend()
    ax.grid(False)
    
    return fig  # Возвращаем фигуру
