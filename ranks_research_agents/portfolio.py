#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numba import njit
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# In[2]:


def calculate_atr(data, period=14):
    """
    Ускоренная версия расчета ATR
    """
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values

    tr = np.empty(len(high))
    tr[0] = high[0] - low[0]

    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    atr = np.empty(len(tr))
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period  # EMA-style smoothing

    return pd.Series(atr, index=data.index)


def extend_signal_hold(signal_bin: np.ndarray, hold: int) -> np.ndarray:
    """Расширение бинарного сигнала на hold дней вперёд.
    Реализовано через свёртку с ядром из единиц, чтобы не гонять Python-цикл.
    signal_bin — 0/1 np.ndarray (int8/uint8). Возвращает int8 0/1.
    """
    if hold <= 1:
        return signal_bin.astype(np.int8, copy=False)
    kernel = np.ones(int(hold), dtype=np.int8)
    # Отрезаем до исходной длины (эквивалент непериодическому "перекатыванию")
    conv = np.convolve(signal_bin.astype(np.int8, copy=False), kernel, mode='full')[: signal_bin.size]
    return (conv > 0).astype(np.int8)

    
# Коды для статусов:
# 0 - 'no action'
# 1 - 'buy'
# 2 - 'hold'
# 3 - 'hold (refresh)'
# 4 - 'sell'

# Коды для позиций:
# 0 - 'out'
# 1 - 'in'

# Коды для причин выхода:
# 0 - ''
# 1 - 'TP'
# 2 - 'SL'
# 3 - 'signal'

@njit
def _calculate_portfolio_numba(
    signal,
    signal_refresh,
    open_prices,
    high_prices,
    low_prices,
    close_prices,
    atr_values,
    SL_mult,
    TP_mult,
    SL_min_pct,
    SL_max_pct,
    sl_chg_thr
):
    """
    СТОП V1
    """
    n = len(signal)
    r_p = np.zeros(n)
    status = np.zeros(n, dtype=np.int32)
    exit_reason = np.zeros(n, dtype=np.int32)
    position = np.zeros(n, dtype=np.int32)
    entry_price_arr = np.full(n, np.nan)
    current_SL_arr = np.full(n, np.nan)
    current_TP_arr = np.full(n, np.nan)

    pending_entry = False
    entry_price = np.nan
    current_SL = np.nan
    current_TP = np.nan
    trade_active = False

    for idx in range(n):
        sig = signal[idx]
        sig_refresh_val = signal_refresh[idx]
        o = open_prices[idx]
        h = high_prices[idx]
        l = low_prices[idx]
        c = close_prices[idx]
        atr = atr_values[idx]

        prev_close = close_prices[idx - 1] if idx > 0 else np.nan

        # Шаг 1: выполнение отложенного входа
        if pending_entry:
            entry_price = c
            sl_raw = SL_mult * atr
            sl_abs = min(max(sl_raw, SL_min_pct * entry_price), SL_max_pct * entry_price)
            current_SL = entry_price - sl_abs
            current_TP = entry_price + TP_mult * atr
        
            # Определяем статус в зависимости от сигнала обновления
            if sig_refresh_val == 1:
                status[idx] = 3  # 'hold (refresh)'
            else:
                status[idx] = 2  # 'hold'
        
            trade_active = True
            pending_entry = False
        
            position[idx] = 1  # 'in'
            entry_price_arr[idx] = entry_price
            current_SL_arr[idx] = current_SL
            current_TP_arr[idx] = current_TP
            r_p[idx] = 0.0
            continue

        # Шаг 2: открытие новой позиции
        if not trade_active and sig_refresh_val == 1:
            # Расчет SL/TP для отображения в день buy
            entry_price_for_calc = close_prices[idx]  # используем текущую цену закрытия для расчета
            sl_raw = SL_mult * atr
            sl_abs = min(max(sl_raw, SL_min_pct * entry_price_for_calc), SL_max_pct * entry_price_for_calc)
            current_SL_temp = entry_price_for_calc - sl_abs
            current_TP_temp = entry_price_for_calc + TP_mult * atr
            
            # Заполняем массивы для текущего дня
            current_SL_arr[idx] = current_SL_temp
            current_TP_arr[idx] = current_TP_temp
            
            status[idx] = 1  # 'buy'
            position[idx] = 0  # 'out'
            r_p[idx] = 0.0
            pending_entry = True
            continue

        # Шаг 3: обработка активной позиции
        if trade_active:
            exit_price = np.nan
            reason = 0

            # --- НОВОЕ ДОП УСЛОВИЕ: гэп ниже вчерашнего SL ---
            # Если цена открытия текущего дня ниже SL,
            # то считаем, что стоп "пробит гэпом",
            # и выходим по цене закрытия (имитация проскальзывания)
            if o < current_SL:
                exit_price = c
                reason = 2  # 'SL'

            # Если гэпа не было — стандартная логика
            elif h >= current_TP:
                exit_price = current_TP
                reason = 1  # 'TP'
            elif l <= current_SL:
                exit_price = current_SL
                reason = 2  # 'SL'
            elif sig == 0:
                exit_price = c
                reason = 3  # 'signal'

            # Обновление SL/TP
            if np.isnan(exit_price) and sig_refresh_val == 1:
                entry_price = c
            
                sl_raw = SL_mult * atr
                sl_abs = min(max(sl_raw, SL_min_pct * entry_price), SL_max_pct * entry_price)
                new_SL = entry_price - sl_abs
            
                # --- НОВАЯ ЛОГИКА ПОРОГА ---
                if np.isnan(current_SL) or abs(new_SL / current_SL - 1) >= sl_chg_thr:
                    current_SL = new_SL
                # иначе оставляем старый SL
            
                # TP обновляем как раньше (без порога)
                current_TP = entry_price + TP_mult * atr
                
                status[idx] = 3  # 'hold (refresh)'
                position[idx] = 1  # 'in'
                entry_price_arr[idx] = entry_price
                current_SL_arr[idx] = current_SL
                current_TP_arr[idx] = current_TP
                r_p[idx] = (c / prev_close - 1) if not np.isnan(prev_close) else 0.0
                continue

            # Выход
            if not np.isnan(exit_price):
            
                if not np.isnan(prev_close):
                    ret = exit_price / prev_close - 1
                else:
                    ret = 0.0
            
                r_p[idx] = ret
                status[idx] = 4  # 'sell'
                exit_reason[idx] = reason
                position[idx] = 0  # 'out'
                entry_price_arr[idx] = entry_price
                current_SL_arr[idx] = current_SL
                current_TP_arr[idx] = current_TP
            
                trade_active = False
                entry_price = np.nan
                current_SL = np.nan
                current_TP = np.nan
                continue

            # Просто держим позицию
            status[idx] = 2  # 'hold'
            position[idx] = 1  # 'in'
            entry_price_arr[idx] = entry_price
            current_SL_arr[idx] = current_SL
            current_TP_arr[idx] = current_TP
            r_p[idx] = (c / prev_close - 1) if not np.isnan(prev_close) else 0.0
            continue

        # Вне позиции
        r_p[idx] = 0.0
        status[idx] = 0  # 'no action'
        position[idx] = 0  # 'out'

    return r_p, status, exit_reason, position, entry_price_arr, current_SL_arr, current_TP_arr


def calculate_portfolio(backtest_df: pd.DataFrame,
                        SL_mult: float = 1.0,
                        TP_mult: float = 2.0,
                        SL_min_pct: float = 0.05,
                        SL_max_pct: float = 0.25,
                        sl_chg_thr: float = 0.02) -> pd.DataFrame:
    df = backtest_df.copy().reset_index(drop=True)

    result = _calculate_portfolio_numba(
        signal=df['signal'].values,
        signal_refresh=df['signal_refresh'].values,
        open_prices=df['Open'].values,
        high_prices=df['High'].values,
        low_prices=df['Low'].values,
        close_prices=df['Close'].values,
        atr_values=df['atr'].values,
        SL_mult=SL_mult,
        TP_mult=TP_mult,
        SL_min_pct=SL_min_pct,
        SL_max_pct=SL_max_pct,
        sl_chg_thr=sl_chg_thr
    )

    df['r_p'], status_codes, exit_reason_codes, position_codes, \
    df['entry_price'], df['current_SL'], df['current_TP'] = result

    # Преобразуем числовые коды обратно в строки
    status_map = {
        0: 'no action',
        1: 'buy',
        2: 'hold',
        3: 'hold (refresh)',
        4: 'sell'
    }
    
    position_map = {
        0: 'out',
        1: 'in'
    }
    
    exit_reason_map = {
        0: '',
        1: 'TP',
        2: 'SL',
        3: 'signal'
    }
    
    df['status'] = [status_map[code] for code in status_codes]
    df['position'] = [position_map[code] for code in position_codes]
    df['exit_reason'] = [exit_reason_map[code] for code in exit_reason_codes]

    return df

