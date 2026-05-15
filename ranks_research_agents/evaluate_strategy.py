#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# In[3]:


def evaluate_strategy(data, freq=252):

    def extract_trades(data):

        trades = []

        # Определяем состояние позиции
        is_in_position = False
        buy_date = None
        buy_index = None

        for i in range(len(data)):
            current_return = data.iloc[i]['r_p']
            current_date = data.iloc[i]['Date']
            current_price = data.iloc[i]['Close']

            # Начало сделки: переход из 0 в ненулевое значение
            if not is_in_position and current_return != 0:
                is_in_position = True
                buy_date = current_date
                buy_price = current_price
                buy_index = i

            # Конец сделки: переход из ненулевого значения в 0
            elif is_in_position and current_return == 0:
                is_in_position = False

                # Рассчитываем доходность сделки
                period_returns = data.iloc[buy_index:i+1]['r_p']
                trade_return = (1 + period_returns).prod() - 1

                # Рассчитываем дни в позиции
                days_in_position = (current_date - buy_date).days

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': current_date,
                    'return': trade_return,
                    'days_in_position': days_in_position,
                    'buy_price': buy_price,
                    'sell_price': current_price
                })

        # Проверяем последнюю открытую позицию
        if is_in_position:
            period_returns = data.iloc[buy_index:]['r_p']
            trade_return = (1 + period_returns).prod() - 1
            days_in_position = (data.iloc[-1]['Date'] - buy_date).days

            trades.append({
                'buy_date': buy_date,
                'sell_date': data.iloc[-1]['Date'],
                'return': trade_return,
                'days_in_position': days_in_position,
                'buy_price': buy_price,
                'sell_price': current_price
            })

        return pd.DataFrame(trades)

    cum_return = (data['r_p'] + 1).prod() - 1
    cum_return_benchmark = (data['r_m'] + 1).prod() - 1

    cumulative_returns = (1 + data['r_p']).cumprod()  # кумулятивная доходность
    running_max = cumulative_returns.expanding().max()  # скользящий максимум
    drawdown = (cumulative_returns - running_max) / running_max  # просадка
    max_drawdown = drawdown.min()  # максимальная просадка

    cumulative_returns_benchmark = (1 + data['r_m']).cumprod()  # кумулятивная доходность
    running_max_benchmark = cumulative_returns_benchmark.expanding().max()  # скользящий максимум
    drawdown_benchmark = (cumulative_returns_benchmark - running_max_benchmark) / running_max_benchmark  # просадка
    max_drawdown_benchmark = drawdown_benchmark.min()  # максимальная просадка

    mean_year_ret = data.groupby(data['Date'].dt.year)['r_p'].apply(lambda x: (x + 1).prod() - 1).mean()
    median_year_ret = data.groupby(data['Date'].dt.year)['r_p'].apply(lambda x: (x + 1).prod() - 1).median()

    mean_benchmark_year_ret = data.groupby(data['Date'].dt.year)['r_m'].apply(lambda x: (x + 1).prod() - 1).mean()
    median_benchmark_year_ret = data.groupby(data['Date'].dt.year)['r_m'].apply(lambda x: (x + 1).prod() - 1).median()

    total_return = (data['r_p'] + 1).prod()
    years = len(data) / freq
    cagr = total_return ** (1 / years) - 1

    calmar = cagr / abs(max_drawdown)

    total_return_benchmark = (data['r_m'] + 1).prod()
    cagr_benchmark = total_return_benchmark ** (1 / years) - 1

    year_vol = data['r_p'].std() * np.sqrt(freq)
    year_vol_benchmark = data['r_m'].std() * np.sqrt(freq)

    sharpe = (data['r_p'] - data['r_f']).mean() * np.sqrt(freq) / data['r_p'].std()

    sortino = (data['r_p'] - data['r_f']).mean() * np.sqrt(freq) / data[data['r_p'] < 0]['r_p'].std()

    beta = data['r_m'].cov(data['r_p']) / data['r_m'].var()
    alpha = (((data['r_p'] - data['r_f']).mean() - beta * (data['r_m'] - data['r_f']).mean()) + 1) ** freq - 1

    trades = extract_trades(data)
    trades['in_cash_return'] = trades['buy_price'].shift(-1) / trades['sell_price'] - 1

    err_1_type = len(trades[trades['return'] < 0]) / len(trades)
    err_2_type = len(trades[trades['in_cash_return'] > 0]) / len(trades)

    mean_trade_return = trades['return'].mean()
    median_trade_return = trades['return'].median()
    mean_in_cash_return = trades['in_cash_return'].mean()
    median_in_cash_return = trades['in_cash_return'].median()

    mean_hold_time = trades['days_in_position'].mean()
    median_hold_time = trades['days_in_position'].median()

    pos_trades_mean_ret = trades[trades['return'] > 0]['return'].mean()
    pos_trades_median_ret = trades[trades['return'] > 0]['return'].median()
    pos_trades_mean_hold = trades[trades['return'] > 0]['days_in_position'].mean()
    pos_trades_median_hold = trades[trades['return'] > 0]['days_in_position'].median()

    neg_trades_mean_ret = trades[trades['return'] <= 0]['return'].mean()
    neg_trades_median_ret = trades[trades['return'] <= 0]['return'].median()
    neg_trades_mean_hold = trades[trades['return'] <= 0]['days_in_position'].mean()
    neg_trades_median_hold = trades[trades['return'] <= 0]['days_in_position'].median()

    market_time = len(data[data['r_p'] != 0]) / len(data)

    results = pd.DataFrame({
        'Портфельные метрики': '------------------------',
        'sharpe': [sharpe],
        'sortino': [sortino],
        'calmar': [calmar],
        'alpha': [alpha],
        'beta': [beta],
        'Годовая волатильность': [year_vol],
        'Годовая волатильность benchmark': [year_vol_benchmark],
        'Максимальная просадка': [max_drawdown],
        'Максимальная просадка benchmark': [max_drawdown_benchmark], 
        'Среднегодовая дох-ть': [mean_year_ret],
        'Медианная годовая дох-ть': [median_year_ret],
        'CAGR': [cagr],
        'Среднегодовая дох-ть benchmark': [mean_benchmark_year_ret],
        'Медианная годовая дох-ть benchmark': [median_benchmark_year_ret],
        'CAGR benchmark': [cagr_benchmark],
        'Накопленная дох-ть': [cum_return],
        'Накопленная дох-ть benchmark': [cum_return_benchmark],
        'Время в рынке': [market_time],
        'Статистика сделок': '------------------------',
        'Кол-во сделок': [int(len(trades))], 
        'Доля отрицательных сделок (ошибка 1 рода)': [err_1_type],
        'Доля выходов в кэш, когда benchmark рос (ошибка 2 рода)': [err_2_type],
        'Средняя дох-ть сделки': [mean_trade_return],
        'Медианная дох-ть сделки': [median_trade_return],
        'Средняя дох-ть benchmark, когда стратегия в кэше': [mean_in_cash_return],
        'Медианная дох-ть benchmark, когда стратегия в кэше': [median_in_cash_return],
        'Среднее время удержания': [np.round(mean_hold_time, 0)], 
        'Медианное время удержания': [np.round(median_hold_time, 0)],
        'Статистика положительных сделок': '------------------------',
        'Средняя дох-ть "+" сделки': [pos_trades_mean_ret],
        'Медианная дох-ть "+" сделки': [pos_trades_median_ret],
        'Среднее время удержания "+" сделки': [np.round(pos_trades_mean_hold, 0)], 
        'Медианное время удержания "+" сделки': [np.round(pos_trades_median_hold, 0)],
        'Статистика отрицательных сделок': '------------------------',
        'Средняя дох-ть "-" сделки': [neg_trades_mean_ret],
        'Медианная дох-ть "-" сделки': [neg_trades_median_ret],
        'Среднее время удержания "-" сделки': [np.round(neg_trades_mean_hold, 0)], 
        'Медианное время удержания "-" сделки': [np.round(neg_trades_median_hold, 0)],
    }, index=[f'backtest: {data["Date"].min().year}-{data["Date"].max().year}']).T.round(4)

    return results

