import importlib.util
import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Функция для загрузки файла
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as file:
        file.write(response.content)
        
# Динамический импорт загруженного модуля
def import_module(module_name):
    spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return globals().update({k: v for k, v in vars(module).items() if not k.startswith('__')})

app_backend_url = "https://raw.githubusercontent.com/dmitriypatlasov/ai-driven-financial-analysis/main/volatility_trading_app/app_backend.py"
download_file(app_backend_url, "app_backend.py")
app_backend_module = import_module("app_backend")

scaler_spy_lstm = "https://github.com/dmitriypatlasov/ai-driven-financial-analysis/raw/main/volatility_trading_app/scaler_spy_lstm.pkl"
download_file(scaler_spy_lstm, "scaler_spy_lstm.pkl")
scaler_spy_lstm = joblib.load('scaler_spy_lstm.pkl')

scaler_qqq_lstm = "https://github.com/dmitriypatlasov/ai-driven-financial-analysis/raw/main/volatility_trading_app/scaler_qqq_lstm.pkl"
download_file(scaler_qqq_lstm, "scaler_qqq_lstm.pkl")
scaler_qqq_lstm = joblib.load('scaler_qqq_lstm.pkl')

spy_lstm_model_url = "https://github.com/dmitriypatlasov/ai-driven-financial-analysis/raw/main/volatility_trading_app/spy_lstm_model.keras"
download_file(spy_lstm_model_url, "spy_lstm_model.keras")
spy_lstm_model = tf.keras.models.load_model('spy_lstm_model.keras')

qqq_lstm_model_url = "https://github.com/dmitriypatlasov/ai-driven-financial-analysis/raw/main/volatility_trading_app/qqq_lstm_model.keras"
download_file(qqq_lstm_model_url, "qqq_lstm_model.keras")
qqq_lstm_model = tf.keras.models.load_model('qqq_lstm_model.keras')

# Основная функция для отображения графиков
def main():
    st.title("Анализ и прогнозирование реализованной волатильности, сигналы к покупке опционов SPY и QQQ")
    current_date = datetime.now().strftime("%Y-%m-%d")
    st.write(f"Дата: {current_date}")
    
    data_spy = get_option_data('SPY')
    data_qqq = get_option_data('QQQ')
    
    recovered_data_spy, pdf_recovered_spy = get_recovered_density(data_spy, 'SPY')
    recovered_data_qqq, pdf_recovered_qqq = get_recovered_density(data_qqq, 'QQQ')
    
    copula_results, spy_states, qqq_states, observe_date = create_copula(pdf_recovered_spy, pdf_recovered_qqq)
    st.subheader("Визуализация копулы t-Стьюдента для ожидаемой доходности SPY и QQQ")
    figure = copula_plot(spy_states, qqq_states, observe_date)
    if isinstance(figure, go.Figure):
        st.plotly_chart(figure)
    else:
        st.error("Ошибка при создании графика копулы.")

    spy_rv_predictions, qqq_rv_predictions, spy_forecast_data, qqq_forecast_data, spy_data, qqq_data = lstm_forecast(
        copula_results, 
        spy_lstm_model, 
        qqq_lstm_model, 
        scaler_spy_lstm, 
        scaler_qqq_lstm
    )
    st.subheader("Прогноз реализованной волатильности SPY и QQQ")
    fig_spy, fig_qqq = plot_forecast(spy_data, spy_rv_predictions, qqq_data, qqq_rv_predictions, observe_date)
    if isinstance(fig_spy, go.Figure):
        st.plotly_chart(fig_spy, key="fig_spy")
    else:
        st.error("Ошибка при создании графика прогноза SPY.")

    if isinstance(fig_qqq, go.Figure):
        st.plotly_chart(fig_qqq, key="fig_qqq")
    else:
        st.error("Ошибка при создании графика прогноза QQQ.")
    
    st.subheader("Возможности для открытия позиций в опционах SPY и QQQ")
    
    option_trading_recomendation(spy_forecast_data, qqq_forecast_data, spy_rv_predictions, qqq_rv_predictions, observe_date)

    # Обновляем состояние расчета
    st.session_state['calculation_done'] = True

if __name__ == "__main__":
    main()