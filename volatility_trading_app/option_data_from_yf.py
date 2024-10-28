#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime as dt

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
                'volume': options_data.option_chain(exp_date).calls['volume'],
            }).assign(Tenor=(pd.to_datetime(exp_date) - today).days / 365)
            for exp_date in expiration_dates
        ],
        ignore_index=True
    )

    # Фильтруем данные
    data = data[(data['Tenor'] <= 1) & (data['volume'] > 500)]
    data = data.drop(columns='volume')

    return data

