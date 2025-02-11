import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal, gaussian_kde, multivariate_t
import yfinance as yf
from datetime import timedelta, datetime
from datetime import datetime as dt
from datetime import date as dt_date
from scipy.linalg import eig, solve
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import tensorflow as tf
import joblib

def get_option_data(instrument):
    """
    Получает данные по опционам для заданного тикера и возвращает DataFrame.

    Параметры:
    ----------
    instrument : str
        Тикер актива (например, 'SPY').

    Возвращает:
    -----------
    pd.DataFrame
        DataFrame с данными по опционам, содержащий столбцы: STRIKE, C_IV, expiration_date, Tenor.
    """
    
    # Получаем данные по тикеру
    options_data = yf.Ticker(instrument)

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
                'expiration_date': pd.to_datetime(exp_date),
            }).assign(Tenor=(pd.to_datetime(exp_date) - today).days / 365)
            for exp_date in expiration_dates
        ],
        ignore_index=True
    )

    # Фильтруем данные
    data = data[(data['Tenor'] <= 1) & (data['Tenor'] >= 0)]

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

def calculate_p_matrix(state_prices):
    """
    Рассчитывает матрицу вероятностей перехода p_matrix для заданных значений state_prices.

    Параметры:
    state_prices (pd.DataFrame): DataFrame с ценами состояний, где строки и столбцы соответствуют состояниям.

    Возвращает:
    pd.DataFrame: Матрица вероятностей перехода p_matrix, нормированная и подготовленная.
    """
    state_prices_values = state_prices.values
    size = len(state_prices)

    def error_function(p_values):
        """Целевая функция для минимизации."""
        p_matrix = np.array(p_values).reshape((size, size))

        total_error = 0
        for i in range(size):
            row_error = 0
            for j in range(1, size + 1):
                state_prices_hat_ij = np.dot(state_prices_values[i, 1:], p_matrix[:, i])
                row_error += (state_prices_hat_ij - state_prices_values[i][j - 1]) ** 2
            total_error += row_error
        return total_error

    def create_ordering_constraints(size):
        """Создает список ограничений для матрицы вероятностей перехода размером size x size."""
        ordering_constraints = []

        for i in range(size):
            for j in range(size):
                if j < i:
                    # Условие для элементов до диагонали: возрастание
                    ordering_constraints.append({
                        'type': 'ineq',
                        'fun': lambda p_values, i=i, j=j: p_values[size * i + j] - p_values[size * i + j - 1]
                    })
                elif j > i:
                    # Условие для элементов после диагонали: убывание
                    ordering_constraints.append({
                        'type': 'ineq',
                        'fun': lambda p_values, i=i, j=j: p_values[size * i + j] - p_values[size * i + j + 1]
                    })

            # Ограничение для строгого максимального значения на диагонали
            max_index = size * i + i
            row_indices = [size * i + k for k in range(size) if k != i]
            ordering_constraints.append({
                'type': 'eq',
                'fun': lambda p_values, max_index=max_index, row_indices=row_indices:
                    p_values[max_index] - max([p_values[idx] for idx in row_indices]) - 1e-8
            })

        return ordering_constraints

    # Задаем ограничения на суммы строк
    row_sum_constraints = {
        'type': 'eq',
        'fun': lambda p_values: np.array([p_values[i:i + size].sum() - 1 for i in range(0, size**2, size)])
    }

    # Создаем начальное предположение и ограничения
    initial_guess = np.full((size, size), 0).flatten()
    constraints = [row_sum_constraints] + create_ordering_constraints(size)
    bounds = [(0, 1) for _ in range(size * size)]

    # Запускаем оптимизацию
    result = minimize(error_function, initial_guess, bounds=bounds, constraints=constraints, method='Nelder-Mead')

    # Преобразуем результат в матрицу и корректируем центральную строку
    p_matrix = result.x.reshape((size, size))
    central_column_values = [row[0] for row in state_prices_values]
    central_row_index = size // 2
    p_matrix[central_row_index] = central_column_values

    # Создаем DataFrame и нормируем значения
    p_matrix = pd.DataFrame(p_matrix, index=state_prices.index, columns=state_prices.index)
    p_matrix[p_matrix < 0] = 0
    p_matrix = p_matrix.div(p_matrix.sum(axis=1), axis=0)

    return p_matrix

def calculate_f_matrix(p_matrix, state_prices):
    """
    Реализация Теоремы восстановления для вычисления матрицы натуральных вероятностей F.
    
    Параметры:
    - P: numpy.ndarray, матрица риск-нейтральных вероятностей (m x m), строки суммируются к 1.
    
    Возвращает:
    - F: numpy.ndarray, матрица натуральных вероятностей (m x m).
    - delta: float, дисконтирующий фактор.
    - z: numpy.ndarray, собственный вектор, связанный с delta.
    """
    # Шаг 1: Находим максимальный собственный вектор и собственное значение для матрицы P
    eigenvalues, eigenvectors = np.linalg.eig(p_matrix.T)  # используем транспонированную матрицу для поиска левого собственного вектора
    idx = np.argmax(np.real(eigenvalues))  # находим индекс максимального собственного значения
    delta = np.real(eigenvalues[idx])  # это значение соответствует дисконтирующему фактору
    z = np.real(eigenvectors[:, idx])  # собственный вектор, связанный с delta
    z = z / np.sum(z)  # нормализуем собственный вектор так, чтобы суммы элементов были равны 1
    
    # Шаг 2: Построение диагональной матрицы D на основе z
    D = np.diag(1 / z)  # диагональная матрица с элементами 1/z_i
    
    # Шаг 3: Вычисляем матрицу натуральных вероятностей F
    f_matrix = (1 / delta) * D @ p_matrix @ np.linalg.inv(D)
    f_matrix.set_index(state_prices.index, inplace=True)
    f_matrix.columns = state_prices.index
    f_matrix = f_matrix.div(f_matrix.sum(axis=1), axis=0)
    
    return f_matrix, delta, z

def distribution_from_markov_chain(markov_chain_matrix):
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
    
    # Получаем состояния из индекса матрицы
    states = markov_chain_matrix.index

    # Решение уравнения πP = π, добавив условие, что сумма π = 1
    A = np.transpose(markov_chain_matrix) - np.eye(markov_chain_matrix.shape[0])
    A = np.vstack([A, np.ones(markov_chain_matrix.shape[0])])
    b = np.zeros(markov_chain_matrix.shape[0] + 1)
    b[-1] = 1  # Сумма равна 1

    # Решаем систему уравнений для нахождения стационарного распределения
    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    # Нормируем стационарное распределение
    pi = pi / np.sum(pi)  # Убедитесь, что сумма равна 1

    # Фильтруем нулевые значения
    non_zero_indices = pi > 0
    pi = pi[non_zero_indices]
    states = states[non_zero_indices]

    return pi, states

def calculate_moments(pi, states):
    mean = np.sum(pi * states) # Вычисление среднего
    
    variance = np.sum(pi * (states - mean)**2) # Вычисление дисперсии и стандартного отклонения
    std_dev = np.sqrt(variance)
    
    skewness = np.sum(pi * (states - mean)**3) / std_dev**3 # Вычисление асимметрии
    
    kurtosis = np.sum(pi * (states - mean)**4) / std_dev**4 - 3 # Вычисление эксцесса
    
    return mean, variance, skewness, kurtosis

def get_recovered_density(data, instrument):
    
    selected_date = pd.to_datetime(dt_date.today())
    undelying_last = (yf.download(instrument, start=(pd.to_datetime(dt_date.today()) - timedelta(days=100)), progress=False)['Close']).iloc[-1][-1]

    dates = []
    risk_neutral_std = []
    risk_neutral_mean = []
    recovered_std = []
    recovered_mean = []
    
    iv_surface = data[['STRIKE', 'C_IV', 'Tenor']].copy().apply(pd.to_numeric, errors='coerce')
    iv_surface = iv_surface.pivot_table(values='C_IV', index='STRIKE', columns='Tenor').dropna()
    tenor = [round((i + 1) / 12, 4) for i in range(12)]
    percent_changes = np.arange(-18, 21, 3) / 100
    strike = undelying_last * (1 + percent_changes)

    # Подготовка данных для интерполяции
    iv_surface_empty = pd.DataFrame(index=strike, columns=tenor) # Создание сетки для интерполяции
    X = iv_surface.columns.astype(float)
    Y = iv_surface.index.astype(float)
    Z = iv_surface.values
    interp_func = RectBivariateSpline(Y, X, Z, kx=1, ky=1)
    empty_matrix_tenor = iv_surface_empty.columns.astype(float)
    empty_matrix_strike = iv_surface_empty.index.astype(float)
    interpolated_values = interp_func(empty_matrix_strike, empty_matrix_tenor, grid=True)
    iv_surface_interpolated = pd.DataFrame(interpolated_values, index=iv_surface_empty.index, columns=iv_surface_empty.columns).dropna()
    iv_surface_interpolated.index.name = 'Strike'
    iv_surface_interpolated.columns.name = 'Tenor'

    ################ Option prices with interpolated implied volatility ################
    ticker = "^TNX"  # Тикер для 10-летних государственных облигаций
    start_date = selected_date - timedelta(days=7)
    end_date = selected_date
    interest_rate_data = yf.download("^TNX", start=start_date, end=end_date, progress=False)
    r = interest_rate_data['Close'].iloc[-1][-1] / 100

    call_prices = pd.DataFrame(index=iv_surface_interpolated.index, columns=iv_surface_interpolated.columns)
    for tenor in iv_surface_interpolated.columns:
        for strike in iv_surface_interpolated.index:
            volatility = iv_surface_interpolated.loc[strike, tenor]
            call_prices.loc[strike, tenor] = black_scholes_call(undelying_last, strike, r, tenor, volatility)

    ################ State price matrix ################

    state_prices = pd.DataFrame(index=call_prices.index, columns=call_prices.columns) # State price matrix (Breeden-Litzenberg) 

    for i in range(1, len(call_prices.index) - 1):
        for j in range(len(call_prices.columns)):
            state_prices.iloc[i, j] = np.exp(r * call_prices.columns[i]) * (
                call_prices.iloc[i + 1, j] - 2 * call_prices.iloc[i, j] + call_prices.iloc[i - 1, j]) / (
                    (call_prices.index[i+1] - call_prices.index[i])**2)

    state_prices = state_prices.iloc[1:-1].copy()
    state_prices[state_prices < 0] = 0

    first_column_sum = state_prices.iloc[:, 0].sum()
    state_prices.iloc[:, 0] = state_prices.iloc[:, 0] / first_column_sum

    central_row = state_prices.index[len(state_prices) // 2]
    percent_change = (state_prices.index - central_row) / central_row
    percent_change = pd.Series(percent_change).round(2)
    state_prices.index = percent_change

    ################ calculate_p_matrix ################

    p_matrix = calculate_p_matrix(state_prices)

    ################ Risk neutral moments ################

    pdf_rn, continious_states_rn = distribution_from_markov_chain(p_matrix)
    pdf_rn_df = pd.DataFrame(pdf_rn.reshape(1, -1), columns=continious_states_rn) # Плотность
    rn_mean, rn_std, rn_skewness, rn_kurtosis = calculate_moments(pdf_rn, continious_states_rn) #  Моменты

    ################ calculate_f_matrix ################

    f_matrix, _, _ = calculate_f_matrix(p_matrix, state_prices)

    ################ Ross recovery moments ################

    pdf_recovered, continious_states_recovered = distribution_from_markov_chain(f_matrix)
    pdf_recovered_df = pd.DataFrame(pdf_recovered.reshape(1, -1), columns=continious_states_recovered) # Плотность
    rec_mean, rec_std, rec_skewness, rec_kurtosis = calculate_moments(pdf_recovered, continious_states_recovered) # Моменты
    
    selected_date = [pd.Timestamp(selected_date)]
    recovered_data = pd.DataFrame({
        'risk_neutral_mean':rn_mean, 
        'risk_neutral_std':rn_std,
        'recovered_mean':rec_mean, 
        'recovered_std':rec_std
    }, index=selected_date)
    
    recovered_data['risk_premium'] = (recovered_data['recovered_mean'] - recovered_data['risk_neutral_mean'])
    recovered_data['risk_preference'] = (recovered_data['recovered_std'] - recovered_data['risk_neutral_std'])

    return recovered_data, pdf_recovered_df

def create_copula(pdf_recovered_spy, pdf_recovered_qqq):
    pdf_combined = pd.concat([pdf_recovered_spy, pdf_recovered_qqq], axis=1, keys=['pdf_recovered_spy', 'pdf_recovered_qqq'], join='inner')

    spy_qqq_returns = yf.download(['SPY', 'QQQ'], start=(pd.to_datetime(dt_date.today()) - timedelta(days=500)), progress=False)['Close']
    spy_qqq_returns = spy_qqq_returns.pct_change().dropna()
    observe_date = spy_qqq_returns.index[-1]
    days_for_stats = 30*6
    
    pdf_to_copula = pd.concat([pdf_combined['pdf_recovered_spy'], pdf_combined['pdf_recovered_qqq']], axis=0).astype(float).T
    pdf_to_copula.index = pdf_to_copula.index.astype(float)
    pdf_to_copula.columns = ['spy_pdf_recovered', 'qqq_pdf_recovered']
    
    # Накопительные функции распределения (CDF)
    pdf_to_copula['cdf_spy'] = pdf_to_copula['spy_pdf_recovered'].cumsum() / pdf_to_copula['spy_pdf_recovered'].sum()
    pdf_to_copula['cdf_qqq'] = pdf_to_copula['qqq_pdf_recovered'].cumsum() / pdf_to_copula['qqq_pdf_recovered'].sum()
    
    # Параметры копулы
    spy_qqq_corr = spy_qqq_returns[(spy_qqq_returns.index > pd.to_datetime(dt_date.today()) - timedelta(days=days_for_stats)) & 
                                   (spy_qqq_returns.index <= pd.to_datetime(dt_date.today()))]
    correlation = spy_qqq_corr['SPY'].corr(spy_qqq_corr['QQQ'])
    cov_matrix = [[1, correlation], [correlation, 1]]
    
    cov_matrix = [[1, correlation], [correlation, 1]]
    
    df = 5  # Степени свободы для t-распределения
    
    # Сгенерируем t-копулу
    num_samples = 500
    copula_samples = multivariate_t.rvs(loc=[0, 0], shape=cov_matrix, df=df, size=num_samples)
    
    # Применим стандартную нормальную CDF для перехода от нормальных квантилей к [0,1]
    u = norm.cdf(copula_samples[:, 0])
    v = norm.cdf(copula_samples[:, 1])
    
    # Преобразуем квантильные значения обратно в состояния SPY и QQQ
    spy_states = np.interp(u, pdf_to_copula['cdf_spy'], pdf_to_copula.index)
    qqq_states = np.interp(v, pdf_to_copula['cdf_qqq'], pdf_to_copula.index)
    
    bootstrap_sample_indices = np.random.choice(len(spy_states), len(spy_states), replace=True)
    bootstrap_spy_states = spy_states[bootstrap_sample_indices]
    bootstrap_qqq_states = qqq_states[bootstrap_sample_indices]
    
    copula_results = pd.DataFrame({
        'date': [spy_qqq_returns.index[-1]],
        'bootstrap_mean_spy': [np.mean(bootstrap_spy_states)],
        'bootstrap_std_spy': [np.std(bootstrap_spy_states)],
        'bootstrap_skew_spy': [skew(bootstrap_spy_states)],
        'bootstrap_kurtosis_spy': [kurtosis(bootstrap_spy_states)],
        'bootstrap_mean_qqq': [np.mean(bootstrap_qqq_states)],
        'bootstrap_std_qqq': [np.std(bootstrap_qqq_states)],
        'bootstrap_skew_qqq': [skew(bootstrap_qqq_states)],
        'bootstrap_kurtosis_qqq': [kurtosis(bootstrap_qqq_states)]
    })

    return copula_results, spy_states, qqq_states, observe_date

def copula_plot(spy_states, qqq_states, observe_date):
    # Преобразуем данные в нужную форму для KDE
    data = np.vstack([spy_states, qqq_states])  # (2, N)
    kde = gaussian_kde(data)

    # Создаем сетку для плотности
    x = np.linspace(spy_states.min(), spy_states.max(), 100)
    y = np.linspace(qqq_states.min(), qqq_states.max(), 100)
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Создаём 3D-график
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="plasma")])
    fig.update_layout(
        title=f"Density Plot of SPY and QQQ States on {observe_date.strftime('%Y-%m-%d')}",
        scene=dict(
            xaxis_title="SPY ETF State",
            yaxis_title="QQQ ETF State",
            zaxis_title="Density"
        )
    )
    
    return fig
    
def lstm_forecast(copula_results, spy_lstm_model, qqq_lstm_model, scaler_spy_lstm, scaler_qqq_lstm):
    def calculate_realized_volatility(data, window):
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['sqrt_returns'] = data['log_returns']**2
        data['rv_t'] = data['sqrt_returns'].rolling(window=window).mean().apply(np.sqrt)
        return data['rv_t']
    
    spy_data = yf.download('SPY', start=(pd.to_datetime(dt_date.today()) - timedelta(days=300)), progress=False)
    spy_data = pd.DataFrame(calculate_realized_volatility(spy_data, window=21)).dropna()
    spy_data['rv_t-5'] = spy_data['rv_t'].rolling(5).mean()
    spy_data['rv_t-21'] = spy_data['rv_t'].rolling(21).mean()
    spy_data.dropna(inplace=True)
    
    qqq_data = yf.download('QQQ', start=(pd.to_datetime(dt_date.today()) - timedelta(days=300)), progress=False)
    qqq_data = pd.DataFrame(calculate_realized_volatility(qqq_data, window=21)).dropna()
    qqq_data['rv_t-5'] = qqq_data['rv_t'].rolling(5).mean()
    qqq_data['rv_t-21'] = qqq_data['rv_t'].rolling(21).mean()
    qqq_data.dropna(inplace=True)
    
    spy_forecast_data = copula_results[['bootstrap_mean_spy', 'bootstrap_std_spy', 'bootstrap_skew_spy', 'bootstrap_kurtosis_spy']]
    spy_forecast_data.columns = ['mean', 'std', 'skewness', 'kurtosis']
    spy_forecast_data['rv_t'] = spy_data['rv_t'][-1]
    spy_forecast_data['rv_t-5'] = spy_data['rv_t-5'][-1]
    spy_forecast_data['rv_t-21'] = spy_data['rv_t-21'][-1]
    
    qqq_forecast_data = copula_results[['bootstrap_mean_qqq', 'bootstrap_std_qqq', 'bootstrap_skew_qqq', 'bootstrap_kurtosis_qqq']]
    qqq_forecast_data.columns = ['mean', 'std', 'skewness', 'kurtosis']
    qqq_forecast_data['rv_t'] = qqq_data['rv_t'][-1]
    qqq_forecast_data['rv_t-5'] = qqq_data['rv_t-5'][-1]
    qqq_forecast_data['rv_t-21'] = qqq_data['rv_t-21'][-1]
    
    spy_forecast_scaled_data = scaler_spy_lstm.transform(spy_forecast_data)
    qqq_forecast_scaled_data = scaler_qqq_lstm.transform(qqq_forecast_data)
    
    spy_forecast_scaled_data = spy_forecast_scaled_data.reshape(spy_forecast_scaled_data.shape[0], 1, spy_forecast_scaled_data.shape[1])
    qqq_forecast_scaled_data = qqq_forecast_scaled_data.reshape(qqq_forecast_scaled_data.shape[0], 1, qqq_forecast_scaled_data.shape[1])
    
    spy_rv_predictions = spy_lstm_model.predict(spy_forecast_scaled_data)
    qqq_rv_predictions = qqq_lstm_model.predict(qqq_forecast_scaled_data)

    return spy_rv_predictions, qqq_rv_predictions, spy_forecast_data, qqq_forecast_data, spy_data, qqq_data

def plot_forecast(spy_data, spy_rv_predictions, qqq_data, qqq_rv_predictions, observe_date):
    # график rv spy
    step = 5
    # Получение столбца с реализованной волатильностью
    realized_volatility = spy_data['rv_t']
    # Определение последнего индекса и расчет будущих индексов с шагом step
    last_index = realized_volatility.index[-1]
    future_indices = [last_index + pd.DateOffset(days=i * step) for i in range(1, spy_rv_predictions.shape[1] + 1)]
    # Преобразуем spy_rv_predictions в одномерный массив
    spy_rv_predictions = spy_rv_predictions.flatten()
    # Создаем фигуру
    fig = go.Figure()
    # Линия реализованной волатильности
    fig.add_trace(go.Scatter(
    x=realized_volatility.index,
    y=realized_volatility,
    mode='lines',
    name='Realized Volatility',
    line=dict(color='black', width=2)
    ))
    
    # Точки предсказанных значений
    fig.add_trace(go.Scatter(
    x=future_indices,
    y=spy_rv_predictions,
    mode='markers',
    name='Predicted Volatility',
    marker=dict(color='red', size=8),
    hoverinfo='x+y'
    ))
    
    # Соединяющая линия прогнозов
    x_values = [last_index] + future_indices
    y_values = [realized_volatility.iloc[-1]] + spy_rv_predictions.tolist()
    
    fig.add_trace(go.Scatter(
    x=x_values,
    y=y_values,
    mode='lines',
    name='Forecast Line',
    line=dict(color='coral', width=2, dash='dash')
    ))
    
    # Настройка осей
    fig.update_layout(
    width=1000,  # Ширина фигуры
    height=500,  # Высота фигуры
    title=f"SPY Realized Volatility Forecast {observe_date.strftime('%Y-%m-%d')}",
    xaxis=dict(
        title="Date",
        tickformat='%b %d, %Y',  # Формат дат: "Jan 10, 2025"
        showgrid=True
    ),
    yaxis=dict(
        title="Realized Volatility, σ",
        showgrid=True
    ),
    legend=dict(
        x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'
    )
    )
    # Показать график
    fig.show()
    
    # график rv qqq
    realized_volatility = qqq_data['rv_t']
    last_index = realized_volatility.index[-1]
    future_indices = [last_index + pd.DateOffset(days=i * step) for i in range(1, qqq_rv_predictions.shape[1] + 1)]
    qqq_rv_predictions = qqq_rv_predictions.flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=realized_volatility.index,
    y=realized_volatility,
    mode='lines',
    name='Realized Volatility',
    line=dict(color='black', width=2)
    ))
    fig.add_trace(go.Scatter(
    x=future_indices,
    y=qqq_rv_predictions,
    mode='markers',
    name='Predicted Volatility',
    marker=dict(color='red', size=8),
    hoverinfo='x+y'
    ))
    x_values = [last_index] + future_indices
    y_values = [realized_volatility.iloc[-1]] + qqq_rv_predictions.tolist()
    fig.add_trace(go.Scatter(
    x=x_values,
    y=y_values,
    mode='lines',
    name='Forecast Line',
    line=dict(color='coral', width=2, dash='dash')
    ))
    fig.update_layout(
    width=1000,  # Ширина фигуры
    height=500,  # Высота фигуры
    title=f"QQQ Realized Volatility Forecast from {observe_date.strftime('%Y-%m-%d')}",
    xaxis=dict(
        title="Date",
        tickformat='%b %d, %Y',  # Формат дат: "Jan 10, 2025"
        showgrid=True
    ),
    yaxis=dict(
        title="Realized Volatility, σ",
        showgrid=True
    ),
    legend=dict(
        x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'
    )
    )
    fig.show()

def option_trading_recomendation(spy_forecast_data, qqq_forecast_data, spy_rv_predictions, qqq_rv_predictions, observe_date):
    spy_forecast_data['rv_pred_t+5'] = spy_rv_predictions[0, 0]
    spy_forecast_data['rv_pred_t+10'] = spy_rv_predictions[0, 1]
    spy_forecast_data['rv_pred_t+15'] = spy_rv_predictions[0, 2]
    spy_forecast_data['rv_pred_t+20'] = spy_rv_predictions[0, 3]
    
    qqq_forecast_data['rv_pred_t+5'] = qqq_rv_predictions[0, 0]
    qqq_forecast_data['rv_pred_t+10'] = qqq_rv_predictions[0, 1]
    qqq_forecast_data['rv_pred_t+15'] = qqq_rv_predictions[0, 2]
    qqq_forecast_data['rv_pred_t+20'] = qqq_rv_predictions[0, 3]
    
    print('SPY')
    print('\nvega-scalping:')
    if spy_forecast_data['rv_pred_t+5'][0] >= spy_forecast_data['rv_t'][0] * 1.05:
        growth = spy_forecast_data['rv_pred_t+5'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"5-дн рост RV по SPY > 5% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(5-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(5+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.65), winrate (44%), ожидаемая доходность (12%), сделок (558), средняя vega (23.40)')
    if spy_forecast_data['rv_pred_t+5'][0] >= spy_forecast_data['rv_t'][0] * 1.1:
        growth = spy_forecast_data['rv_pred_t+5'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"5-дн рост RV по SPY > 10% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(5-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(5+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.62), winrate (44%), ожидаемая доходность (11%), сделок (380), средняя vega (23.67)')
    if spy_forecast_data['rv_pred_t+5'][0] >= spy_forecast_data['rv_t'][0] * 1.15:
        growth = spy_forecast_data['rv_pred_t+5'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"5-дн рост RV по SPY > 15% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(5-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(5+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.82), winrate (45%), ожидаемая доходность (15%), сделок (198), средняя vega (23.58)')
    if spy_forecast_data['rv_pred_t+10'][0] >= spy_forecast_data['rv_t'][0] * 1.05:
        growth = spy_forecast_data['rv_pred_t+10'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"10-дн рост RV по SPY > 5% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.65), winrate (47%), ожидаемая доходность (16%), сделок (312), средняя vega (29.95)')
    if spy_forecast_data['rv_pred_t+10'][0] >= spy_forecast_data['rv_t'][0] * 1.1:
        growth = spy_forecast_data['rv_pred_t+10'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"10-дн рост RV по SPY > 10% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.67), winrate (47%), ожидаемая доходность (17%), сделок (208), средняя vega (30.00)')
    if spy_forecast_data['rv_pred_t+10'][0] >= spy_forecast_data['rv_t'][0] * 1.15:
        growth = spy_forecast_data['rv_pred_t+10'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"10-дн рост RV по SPY > 15% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.69), winrate (48%), ожидаемая доходность (16%), сделок (124), средняя vega (30.04)')
    if spy_forecast_data['rv_pred_t+10'][0] <= spy_forecast_data['rv_t'][0] * 0.95:
        growth = spy_forecast_data['rv_pred_t+10'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"10-дн падение RV по SPY > 5% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.52), winrate (45%), ожидаемая доходность (13%), сделок (246), средняя vega (29.58)')
    if spy_forecast_data['rv_pred_t+10'][0] <= spy_forecast_data['rv_t'][0] * 0.9:
        growth = spy_forecast_data['rv_pred_t+10'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"10-дн падение RV по SPY > 10% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.61), winrate (44%), ожидаемая доходность (15%), сделок (127), средняя vega (29.17)')
    if spy_forecast_data['rv_pred_t+20'][0] >= spy_forecast_data['rv_t'][0] * 1.05:
        growth = spy_forecast_data['rv_pred_t+20'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по SPY > 5% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.59), winrate (51%), ожидаемая доходность (20%), сделок (251), средняя vega (39.59)')
    if spy_forecast_data['rv_pred_t+20'][0] <= spy_forecast_data['rv_t'][0] * 0.85:
        growth = spy_forecast_data['rv_pred_t+20'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"20-дн падение RV по SPY > 15% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах SPY (наибольшая vega, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.58), winrate (48%), ожидаемая доходность (22%), сделок (238), средняя vega (39.21)')

    print('\ndelta-scalping:')
    if spy_forecast_data['rv_pred_t+10'][0] >= spy_forecast_data['rv_t'][0] * 1.15:
        growth = spy_forecast_data['rv_pred_t+10'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"10-дн рост RV по SPY > 15% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах SPY (наибольшая delta, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.94), winrate (62%), ожидаемая доходность (11%), сделок (123), средняя delta (0.85)')
    if spy_forecast_data['rv_pred_t+20'][0] >= spy_forecast_data['rv_t'][0] * 1.05:
        growth = spy_forecast_data['rv_pred_t+20'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по SPY > 5% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах SPY (наибольшая delta, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.82), winrate (61%), ожидаемая доходность (16%), сделок (246), средняя delta (0.80)')
    if spy_forecast_data['rv_pred_t+20'][0] >= spy_forecast_data['rv_t'][0] * 1.10:
        growth = spy_forecast_data['rv_pred_t+20'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по SPY > 10% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах SPY (наибольшая delta, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.88), winrate (58%), ожидаемая доходность (16%), сделок (172), средняя delta (0.81)')
    if spy_forecast_data['rv_pred_t+20'][0] >= spy_forecast_data['rv_t'][0] * 1.15:
        growth = spy_forecast_data['rv_pred_t+20'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по SPY > 15% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах SPY (наибольшая delta, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.06), winrate (59%), ожидаемая доходность (16%), сделок (115), средняя delta (0.83)')
        
    print('\nstraddle:')
    if spy_forecast_data['rv_pred_t+10'][0] <= spy_forecast_data['rv_t'][0] * 0.9:
        growth = spy_forecast_data['rv_pred_t+10'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"10-дн падение RV по SPY > 10% (прогноз {growth:.2%}), стратегия: straddle на опционах SPY (самые дальние от страйка колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.35), winrate (59%), ожидаемая доходность (4%), сделок (129)')
    if spy_forecast_data['rv_pred_t+15'][0] <= spy_forecast_data['rv_t'][0] * 0.9:
        growth = spy_forecast_data['rv_pred_t+15'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"15-дн падение RV по SPY > 10% (прогноз {growth:.2%}), стратегия: straddle на опционах SPY (самые дальние от страйка колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(15-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(15+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.32), winrate (54%), ожидаемая доходность (5%), сделок (172)')
    if spy_forecast_data['rv_pred_t+20'][0] <= spy_forecast_data['rv_t'][0] * 0.95:
        growth = spy_forecast_data['rv_pred_t+20'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"20-дн падение RV по SPY > 5% (прогноз {growth:.2%}), стратегия: straddle на опционах SPY (самые дальние от страйка колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.35), winrate (56%), ожидаемая доходность (6%), сделок (378)')
    if spy_forecast_data['rv_pred_t+20'][0] <= spy_forecast_data['rv_t'][0] * 0.9:
        growth = spy_forecast_data['rv_pred_t+20'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"20-дн падение RV по SPY > 10% (прогноз {growth:.2%}), стратегия: straddle на опционах SPY (самые дальние от страйка колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.36), winrate (57%), ожидаемая доходность (6%), сделок (305)')
    if spy_forecast_data['rv_pred_t+20'][0] <= spy_forecast_data['rv_t'][0] * 0.85:
        growth = spy_forecast_data['rv_pred_t+20'][0] / spy_forecast_data['rv_t'][0] - 1
        print(f"20-дн падение RV по SPY > 15% (прогноз {growth:.2%}), стратегия: straddle на опционах SPY (самые дальние от страйка колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.32), winrate (57%), ожидаемая доходность (5%), сделок (230)')

    print('\nQQQ')
    print('\nvega-scalping:')
    if qqq_forecast_data['rv_pred_t+5'][0] >= qqq_forecast_data['rv_t'][0] * 1.05:
        growth = qqq_forecast_data['rv_pred_t+5'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"5-дн рост RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах QQQ (наибольшая vega, экспирация опционов от {(observe_date + timedelta(5-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(5+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.37), winrate (52%), ожидаемая доходность (26%), сделок (112), средняя vega (18.21)')
    if qqq_forecast_data['rv_pred_t+10'][0] >= qqq_forecast_data['rv_t'][0] * 1.05:
        growth = qqq_forecast_data['rv_pred_t+10'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"10-дн рост RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах QQQ (наибольшая vega, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.23), winrate (51%), ожидаемая доходность (33%), сделок (174), средняя vega (23.65)')
    if qqq_forecast_data['rv_pred_t+15'][0] >= qqq_forecast_data['rv_t'][0] * 1.05:
        growth = qqq_forecast_data['rv_pred_t+15'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"15-дн рост RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах QQQ (наибольшая vega, экспирация опционов от {(observe_date + timedelta(15-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(15+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.12), winrate (52%), ожидаемая доходность (40%), сделок (207), средняя vega (28.71)')
    if qqq_forecast_data['rv_pred_t+15'][0] >= qqq_forecast_data['rv_t'][0] * 1.1:
        growth = qqq_forecast_data['rv_pred_t+15'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"15-дн рост RV по QQQ > 10% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах QQQ (наибольшая vega, экспирация опционов от {(observe_date + timedelta(15-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(15+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.15), winrate (52%), ожидаемая доходность (39%), сделок (143), средняя vega (29.16)')
    if qqq_forecast_data['rv_pred_t+20'][0] >= qqq_forecast_data['rv_t'][0] * 1.05:
        growth = qqq_forecast_data['rv_pred_t+20'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах QQQ (наибольшая vega, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.09), winrate (54%), ожидаемая доходность (47%), сделок (183), средняя vega (32.92)')
    if qqq_forecast_data['rv_pred_t+20'][0] >= qqq_forecast_data['rv_t'][0] * 1.10:
        growth = qqq_forecast_data['rv_pred_t+20'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по QQQ > 10% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах QQQ (наибольшая vega, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.83), winrate (52%), ожидаемая доходность (31%), сделок (133), средняя vega (33.37)')
    if qqq_forecast_data['rv_pred_t+20'][0] >= qqq_forecast_data['rv_t'][0] * 1.15:
        growth = qqq_forecast_data['rv_pred_t+20'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по QQQ > 15% (прогноз {growth:.2%}), стратегия: vega-scalping на колл-опционах QQQ (наибольшая vega, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.68), winrate (51%), ожидаемая доходность (26%), сделок (103), средняя vega (33.84)')
    
    print('\ndelta-scalping:')
    if qqq_forecast_data['rv_pred_t+5'][0] >= qqq_forecast_data['rv_t'][0] * 1.05:
        growth = qqq_forecast_data['rv_pred_t+5'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"5-дн рост RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах QQQ (наибольшая delta, экспирация опционов от {(observe_date + timedelta(5-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(5+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (2.20), winrate (64%), ожидаемая доходность (18%), сделок (112), средняя delta (0.91)')
    if qqq_forecast_data['rv_pred_t+10'][0] >= qqq_forecast_data['rv_t'][0] * 1.05:
        growth = qqq_forecast_data['rv_pred_t+10'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"10-дн рост RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах QQQ (наибольшая delta, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.87), winrate (67%), ожидаемая доходность (23%), сделок (171), средняя delta (0.85)')
    if qqq_forecast_data['rv_pred_t+15'][0] >= qqq_forecast_data['rv_t'][0] * 1.05:
        growth = qqq_forecast_data['rv_pred_t+15'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"15-дн рост RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах QQQ (наибольшая delta, экспирация опционов от {(observe_date + timedelta(15-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(15+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.33), winrate (66%), ожидаемая доходность (26%), сделок (200), средняя delta (0.8)')
    if qqq_forecast_data['rv_pred_t+15'][0] >= qqq_forecast_data['rv_t'][0] * 1.1:
        growth = qqq_forecast_data['rv_pred_t+15'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"15-дн рост RV по QQQ > 10% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах QQQ (наибольшая delta, экспирация опционов от {(observe_date + timedelta(15-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(15+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.55), winrate (67%), ожидаемая доходность (29%), сделок (138), средняя delta (0.9)')
    if qqq_forecast_data['rv_pred_t+20'][0] >= qqq_forecast_data['rv_t'][0] * 1.05:
        growth = qqq_forecast_data['rv_pred_t+20'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах QQQ (наибольшая delta, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.39), winrate (63%), ожидаемая доходность (35%), сделок (174), средняя delta (0.77)')
    if qqq_forecast_data['rv_pred_t+20'][0] >= qqq_forecast_data['rv_t'][0] * 1.1:
        growth = qqq_forecast_data['rv_pred_t+20'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"20-дн рост RV по QQQ > 10% (прогноз {growth:.2%}), стратегия: delta-scalping на колл-опционах QQQ (наибольшая delta, экспирация опционов от {(observe_date + timedelta(20-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(20+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (1.30), winrate (65%), ожидаемая доходность (31%), сделок (128), средняя delta (0.78)')
    
    print('\nstraddle:')
    if qqq_forecast_data['rv_pred_t+5'][0] <= qqq_forecast_data['rv_t'][0] * 0.9:
        growth = qqq_forecast_data['rv_pred_t+5'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"5-дн падение RV по QQQ > 10% (прогноз {growth:.2%}), стратегия: straddle на опционах QQQ (самые дальние от страйка колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(5-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(5+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.37), winrate (51%), ожидаемая доходность (3%), сделок (102)')
    if qqq_forecast_data['rv_pred_t+10'][0] <= qqq_forecast_data['rv_t'][0] * 0.9:
        growth = qqq_forecast_data['rv_pred_t+10'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"10-дн падение RV по QQQ > 10% (прогноз {growth:.2%}), стратегия: straddle на опционах QQQ (самые дальние от страйка колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.37), winrate (50%), ожидаемая доходность (5%), сделок (224)')
    if qqq_forecast_data['rv_pred_t+10'][0] <= qqq_forecast_data['rv_t'][0] * 0.95:
        growth = qqq_forecast_data['rv_pred_t+10'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"10-дн падение RV по QQQ > 5% (прогноз {growth:.2%}), стратегия: straddle на опционах QQQ (ближайшие к страйку колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.26), winrate (48%), ожидаемая доходность (3%), сделок (400)')
    if qqq_forecast_data['rv_pred_t+10'][0] <= qqq_forecast_data['rv_t'][0] * 0.9:
        growth = qqq_forecast_data['rv_pred_t+10'][0] / qqq_forecast_data['rv_t'][0] - 1
        print(f"10-дн падение RV по QQQ > 10% (прогноз {growth:.2%}), стратегия: straddle на опционах QQQ (ближайшие к страйку колл- и пут-опционы, экспирация опционов от {(observe_date + timedelta(10-3)).strftime('%Y-%m-%d')} до {(observe_date + timedelta(10+3)).strftime('%Y-%m-%d')})")
        print('статистика стратегии: коэф. Шарпа (0.33), winrate (48%), ожидаемая доходность (4%), сделок (224)')
