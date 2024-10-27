#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import yfinance as yf 
import matplotlib.pyplot as plt
from datetime import date as dt_date
from datetime import timedelta

def realized_volatility_calc(instrument, days_back=100, window=21):
    """
    Вычисляет реализованную волатильность на основе логарифмических доходностей.

    Параметры:
    data (pd.DataFrame): Датафрейм, содержащий столбец 'Adj Close' с ценами закрытия.
    window (int): Размер окна для расчета скользящей средней (количество периодов).

    Возвращает:
    pd.Series: Серия реализованной волатильности, рассчитанная по заданному окну.
    """
    
    # Расчет логарифмических доходностей
    
    volatility_df = pd.DataFrame(yf.download(instrument, start=(pd.to_datetime(dt_date.today()) - timedelta(days=days_back)), progress=False)['Adj Close'])
    volatility_df['Log Returns'] = np.log(volatility_df['Adj Close'] / volatility_df['Adj Close'].shift(1))
    
    # Расчет квадратов доходностей
    volatility_df['Squared Returns'] = volatility_df['Log Returns'] ** 2
    
    # Вычисление реализованной волатильности с использованием скользящей средней
    volatility_df['Realized_Volatility_t'] = volatility_df['Squared Returns'].rolling(window=window).mean().apply(np.sqrt)
    
    volatility_df['Realized_Volatility_t-5'] = volatility_df['Realized_Volatility_t'].shift(5)
    volatility_df['Realized_Volatility_t-21'] = volatility_df['Realized_Volatility_t'].shift(21)
    
    return volatility_df.dropna()

def forecast_calc(scaler_features, model, recovered_data, volatility_df):
    inference_features = recovered_data[['recovered_mean', 'recovered_std', 'recovered_skewness', 'recovered_kurtosis', 'risk_preference']]
    realized_volatility_df = volatility_df[['Realized_Volatility_t', 'Realized_Volatility_t-5', 'Realized_Volatility_t-21']]
    inference_features = pd.merge_asof(inference_features, realized_volatility_df.tail(1), left_index=True, right_index=True, direction='backward')
    inference_features = scaler_features.transform(inference_features)
    inference_features_seq = inference_features.reshape(inference_features.shape[0], 1, inference_features.shape[1])

    # Загрузка модели
    prediction = model.predict(inference_features_seq)[0]
    
    return prediction
    
def plot_volatility_forecast(volatility_df, predictions, step=5):
    """
    Визуализирует прогноз реализованной волатильности на основе существующих данных и предсказанных точек.

    Параметры:
    volatility_df (pd.DataFrame): Датафрейм с реализованной волатильностью и ценами акций. 
                                   Ожидается столбец 'Realized_Volatility_t' и 'Adj Close'.
    predictions (list или pd.Series): Прогнозные значения волатильности.
    step (int): Шаг прогноза в днях между предсказанными точками. По умолчанию 5.
    """
    # Получение столбца с реализованной волатильностью и ценой акций
    realized_volatility = volatility_df['Realized_Volatility_t']
    underlying_asset = volatility_df['Adj Close']
    
    # Определение последнего индекса и расчет будущих индексов с заданным шагом
    last_index = realized_volatility.index[-1]
    future_indices = [last_index + pd.DateOffset(days=i * step) for i in range(1, len(predictions) + 1)]
    
    # Построение графика
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Создаем фигуру и оси
    
    # График реализованной волатильности
    ax1.plot(realized_volatility, label='Realized Volatility', color='black', linewidth=2)
    
    # Добавляем предсказанные точки
    ax1.scatter(future_indices, predictions, color='black', label='Predicted Volatility', zorder=10)
    
    # Соединяем конец графика с прогнозными точками одной линией
    x_values = [last_index] + future_indices
    y_values = [realized_volatility.iloc[-1]] + predictions.tolist()
    
    # Проводим линию между последним значением и всеми прогнозами
    ax1.plot(x_values, y_values, color='coral', linestyle='-', linewidth=2, label='Forecast Line')
    
    # Обозначаем точки на графике
    for i, (index, pred) in enumerate(zip(future_indices, predictions), start=1):
        ax1.text(index, pred, f't+{i * step} ', fontsize=10, color='black', ha='right')
    
    # Настройка первой оси
    ax1.set_title("Realized Volatility Forecast")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Realized Volatility", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid(False)

    # Создаем вторую ось для графика underlying_asset
    ax2 = ax1.twinx()  # Создание второй оси Y
    ax2.plot(underlying_asset.index, underlying_asset, label='Underlying Asset', color='gray', linewidth=2, linestyle='--')
    
    # Настройка второй оси
    ax2.set_ylabel("Underlying Asset Price", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')
    
    # Показать график
    plt.show()
    
    return fig  # Возвращаем фигуру

