#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import yfinance as yf 
import matplotlib.pyplot as plt

def realized_volatility_calc(instrument, days_back=100):
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
    volatility_df['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    
    # Расчет квадратов доходностей
    volatility_df['Squared Returns'] = volatility_df['Log Returns'] ** 2
    
    # Вычисление реализованной волатильности с использованием скользящей средней
    volatility_df['Realized_Volatility_t'] = volatility_df['Squared Returns'].rolling(window=window).mean().apply(np.sqrt)
    
    volatility_df['Realized_Volatility_t-5'] = volatility_df['Realized_Volatility_t'].shift(5)
    volatility_df['Realized_Volatility_t-21'] = volatility_df['Realized_Volatility_t'].shift(21)
    volatility_df = volatility_df[['Realized_Volatility_t', 'Realized_Volatility_t-5', 'Realized_Volatility_t-21']]
    
    return volatility_df.dropna()

def forecast_calc(scaler_features, model, recovered_data, volatility_df):
    inference_features = recovered_data[['recovered_mean', 'recovered_std', 'recovered_skewness', 'recovered_kurtosis', 'risk_neutral_std']]
    inference_features = pd.merge_asof(inference_features, realized_volatility_df.tail(1), left_index=True, right_index=True, direction='backward')
    inference_features = scaler_features.transform(inference_features)
    inference_features_seq = inference_features.reshape(inference_features.shape[0], 1, inference_features.shape[1])

    # Загрузка модели
    prediction = model.predict(inference_features_seq)[0]
    
    return prediction
    
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

